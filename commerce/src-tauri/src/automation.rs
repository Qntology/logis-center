use anyhow::anyhow;
use chromiumoxide::{Browser, BrowserConfig};
use fantoccini::ClientBuilder;
use futures::StreamExt;
use std::process::{Command, Stdio};
use std::time::Duration;
use std::path::PathBuf;
use tauri::Emitter;
use std::sync::Arc;
use once_cell::sync::Lazy;
use serde_json::json;
use regex::Regex;
use reqwest::Url;

// Global storage to keep browser alive
pub(crate) static GLOBAL_BROWSER: Lazy<Arc<tokio::sync::Mutex<Option<Arc<Browser>>>>> = Lazy::new(|| {
    Arc::new(tokio::sync::Mutex::new(None))
});

// Driver Port (Only for Firefox/Safari)
const DRIVER_PORT: u16 = 4444;
const CHROME_DEBUG_PORT: u16 = 9222;

#[derive(serde::Serialize)]
pub struct BrowserStatus {
    pub name: String,
    pub is_supported: bool, // Chrome/Edge = Driverless Supported
    pub is_installed: bool,
    pub needs_driver: bool, // Firefox/Safari = True
}

// --- URL Patterns (Ported from JS) ---
const CLIENT_PATTERNS: &[&str] = &[
    "*.cafe24.com", "*.makeshop.co.kr", "admin.godo.co.kr", "*.godo.co.kr", "*.firstmall.kr",
    "admin.sixshop.com", "sixshop.com", "admin.imweb.me", "www.imweb.me", "*.myshopify.com",
    "sell.smartstore.naver.com", "wing.coupang.com", "soffice.11st.co.kr", "scm.gmarket.co.kr",
    "scm.auction.co.kr", "seller.interpark.com", "seller.wemakeprice.com", "sell.ssg.com",
    "marketplus.co.kr", "admin.shopby.co.kr", "creators.kakaomakers.com", "sell.storefarm.naver.com",
    "partner.wemakeprice.com", "activeitzone.com", "demofran.com", "*.demofran.com",
    "cafe24.com", "makeshop.co.kr", "godo.co.kr", "firstmall.kr", "myshopify.com"
];

const ADMIN_PATTERNS: &[&str] = &[
    "*.cafe24.com", "*.makeshop.co.kr", "*.godomall.com", "*.godo.co.kr", "*.firstmall.kr",
    "*.sixshop.com", "*.imweb.me", "*.myshopify.com", "*.shopby.co.kr", "*.wisa.co.kr",
    "*.sellstore.co.kr", "*.squarespace.com", "*.storefarm.naver.com", "*.smartstore.naver.com",
    "*.gmkt.kr", "*.gmarket.co.kr", "*.auction.co.kr", "*.interpark.com", "*.wemakeprice.com",
    "*.ssg.com", "*.coupang.com", "*.11st.co.kr", "*.kakaomakers.com", "*.activeitzone.com", "*.demofran.com",
    "demofran.com", "activeitzone.com"
];

fn is_shop(url: &str, patterns: &[&str]) -> bool {
    let host = if let Ok(parsed_url) = Url::parse(url) {
        parsed_url.host_str().unwrap_or("").to_lowercase()
    } else {
        url.to_lowercase()
    };

    if host.is_empty() { return false; }

    for pattern in patterns {
        let clean_pattern = pattern.to_lowercase();
        let regex_str = format!("^{}$", clean_pattern.replace(".", "\\.").replace("*", ".*"));
        if let Ok(re) = Regex::new(&regex_str) {
            if re.is_match(&host) { return true; }
        }
        
        let root = clean_pattern.replace("*.", "");
        if host == root || host.ends_with(&format!(".{}", root)) {
            return true;
        }
    }
    false
}

pub async fn is_browser_reachable() -> bool {
    tokio::net::TcpStream::connect(format!("127.0.0.1:{}", CHROME_DEBUG_PORT)).await.is_ok()
}

// --- Entry Point ---
pub async fn run_browser_automation(
    browser_type: String,
    url: String,
    script: String,
    app_handle: tauri::AppHandle,
) -> anyhow::Result<String> {
    match browser_type.as_str() {
        "chrome" | "edge" => run_driverless_automation(&browser_type, &url, &script, app_handle).await,
        "firefox" | "safari" => run_driver_automation(&browser_type, &url, &script).await,
        _ => Err(anyhow!("Unknown browser type")),
    }
}

// --- Reconnection Logic ---
pub async fn try_reconnect_existing_browser(app_handle: tauri::AppHandle) -> anyhow::Result<()> {
    println!("[AUTO] Attempting to reconnect to existing browser on port {}...", CHROME_DEBUG_PORT);
    
    if tokio::net::TcpStream::connect(format!("127.0.0.1:{}", CHROME_DEBUG_PORT)).await.is_err() {
        println!("[AUTO] No existing browser detected on port {}.", CHROME_DEBUG_PORT);
        let _ = app_handle.emit("browser-status", "stopped");
        return Ok(());
    }

    let addr = format!("http://127.0.0.1:{}", CHROME_DEBUG_PORT);
    match Browser::connect(addr).await {
        Ok((browser, mut handler)) => {
            println!("[AUTO] Successfully reconnected to existing browser.");
            let browser_arc = Arc::new(browser);
            {
                let mut global = GLOBAL_BROWSER.lock().await;
                *global = Some(browser_arc.clone());
            }
            let _ = app_handle.emit("browser-status", "running");
            spawn_browser_monitor(browser_arc.clone(), app_handle.clone());
            tokio::spawn(async move {
                while let Some(h) = handler.next().await {
                    if let Err(_) = h { break; }
                }
                let mut global = GLOBAL_BROWSER.lock().await; 
                *global = None; 
                
            });
            Ok(())
        },
        Err(e) => {
            println!("[AUTO] Reconnection failed: {}", e);
            let _ = app_handle.emit("browser-status", "stopped");
            Ok(())
        }
    }
}

// Global storage for the last detected browser state
pub struct DetectedState {
    pub url: String,
    pub tab_id: String, 
    pub is_client: bool,
    pub is_admin: bool,
}

pub(crate) static LAST_DETECTED_STATE: Lazy<Arc<tokio::sync::Mutex<DetectedState>>> = Lazy::new(|| {
    Arc::new(tokio::sync::Mutex::new(DetectedState {
        url: String::new(),
        tab_id: String::new(), 
        is_client: false,
        is_admin: false,
    }))
});

fn spawn_browser_monitor(browser: Arc<Browser>, app_handle: tauri::AppHandle) {
    tokio::spawn(async move {
        let mut last_detected_url = String::new();
        let mut last_is_shop = false;
        let mut fail_count = 0; 
        
        
        let mut last_focused_tab_id = String::new();

        loop {
            if crate::utils::is_extraction_stopped() {
                println!("[AUTO] Global stop signal detected. Exiting browser monitor.");
                break;
            }

            let pages = match browser.pages().await {
                Ok(p) => { fail_count = 0; p },
                Err(e) => {
                    let err_msg = e.to_string();
                    if err_msg.contains("receiver is gone") || err_msg.contains("closed") || fail_count > 5 {
                        println!("[AUTO] Browser disconnected. Exiting monitor.");
                        break;
                    }
                    fail_count += 1;
                    tokio::time::sleep(Duration::from_millis(2000)).await;
                    continue; 
                }, 
            };

            let mut active_url = String::new();
            let mut active_tab_id = String::new();
            let mut is_client = false;
            let mut is_admin = false;

            let mut found_focus = false;
            let mut remembered_url = String::new();
            let mut remembered_visible = false;

            
            for page in pages.iter().rev() {
                let script = r#"
                    (function() {
                        try {
                            window.__logis_tab_id = window.__logis_tab_id || Math.random().toString(36).substring(2);
                            return JSON.stringify({
                                id: window.__logis_tab_id,
                                url: window.location.href,
                                focus: document.hasFocus ? document.hasFocus() : false,
                                visible: document.visibilityState === 'visible'
                            });
                        } catch(e) { return null; }
                    })();
                "#;

                
                let eval_result = tokio::time::timeout(Duration::from_millis(300), page.evaluate(script)).await;

                match eval_result {
                    Ok(Ok(res)) => {
                        if let Some(val_str) = res.into_value::<String>().ok() {
                            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&val_str) {
                                let tab_id = json_val.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                let tab_url = json_val.get("url").and_then(|v| v.as_str()).unwrap_or("");
                                let has_focus = json_val.get("focus").and_then(|v| v.as_bool()).unwrap_or(false);

                                
                                if tab_url.starts_with("devtools://") {
                                    continue;
                                }

                                if has_focus {
                                    last_focused_tab_id = tab_id.to_string();
                                    active_tab_id = tab_id.to_string();
                                    active_url = tab_url.to_string();
                                    found_focus = true;
                                    break;
                                }

                                if !last_focused_tab_id.is_empty() && tab_id == last_focused_tab_id {
                                    remembered_url = tab_url.to_string();
                                    remembered_visible = json_val.get("visible").and_then(|v| v.as_bool()).unwrap_or(false);
                                }
                            }
                        }
                    },
                    _ => continue, // 타임아웃이나 에러 발생 시 해당 탭은 무시하고 루프 유지
                }
            }

            
            if !found_focus {
                if !remembered_url.is_empty() && remembered_visible {
                    // 장부에 적힌 탭이 아직 살아있고 화면에 "보이는(visible)" 상태일 때만 유지!
                    // (만약 다른 탭(chrome:// 등)으로 이동했다면 기존 탭은 hidden이 되므로 이 조건을 통과하지 못함)
                    active_url = remembered_url;
                    active_tab_id = last_focused_tab_id.clone();
                } else {
                    // 장부에 적힌 탭이 닫혀버렸거나 백그라운드(hidden)로 밀려났음 -> 장부 초기화
                    last_focused_tab_id.clear();
                    
                    // Fallback 1: 처음 켰거나 탭이 다 닫힌 경우, 화면에 보이는(visible) 첫 번째 탭을 강제 픽업하여 장부에 등록
                    for page in pages.iter().rev() {
                        let script = r#"
                            (function() {
                                try {
                                    window.__logis_tab_id = window.__logis_tab_id || Math.random().toString(36).substring(2);
                                    return JSON.stringify({ id: window.__logis_tab_id, url: window.location.href, visible: document.visibilityState === 'visible' });
                                } catch(e) { return null; }
                            })();
                        "#;
                        if let Ok(Ok(res)) = tokio::time::timeout(Duration::from_millis(300), page.evaluate(script)).await {
                            if let Some(val_str) = res.into_value::<String>().ok() {
                                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&val_str) {
                                    let tab_url = json_val.get("url").and_then(|v| v.as_str()).unwrap_or("");
                                    
                                    
                                    if tab_url.starts_with("devtools://") {
                                        continue;
                                    }

                                    let is_visible = json_val.get("visible").and_then(|v| v.as_bool()).unwrap_or(false);
                                    if is_visible {
                                        let tab_id = json_val.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                        
                                        last_focused_tab_id = tab_id.to_string(); // 장부 각인!
                                        active_tab_id = tab_id.to_string();
                                        active_url = tab_url.to_string();
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    
                    // visible 상태를 읽어올 수 없는 경우, 엉뚱한 백그라운드 탭을 잡지 않도록 
                    // 명시적으로 "about:blank"를 주어 프론트엔드가 즉시 번개 버튼(extract)을 숨기도록 유도합니다!
                    if active_url.is_empty() {
                        active_url = "about:blank".to_string();
                    }
                }
            }

            // URL 판별 로직
            if !active_url.is_empty() && active_url != "about:blank" && !active_url.starts_with("chrome://") && !active_url.starts_with("edge://") {
                is_client = is_shop(&active_url, CLIENT_PATTERNS);
                is_admin = is_shop(&active_url, ADMIN_PATTERNS);
            }

            let current_is_shop = is_client || is_admin;

            // 전역 상태에 탭 ID와 URL 저장
            {
                let mut state = LAST_DETECTED_STATE.lock().await;
                state.url = active_url.clone();
                state.tab_id = active_tab_id.clone();
                state.is_client = is_client;
                state.is_admin = is_admin;
            }

            // UI 통신: URL이 변경되지 않았더라도 브라우저가 물리적으로 살아있다면 
            // 프론트엔드에 "running" 상태와 현재 URL 정보를 매 루프(800ms)마다 강제 동기화합니다.
            // URL이 비어있어도(about:blank) status가 running이면 Launch 버튼은 숨겨져야 합니다.
            let payload = json!({
                "url": active_url.clone(),
                "is_client": is_client,
                "is_admin": is_admin,
                "status": "running",
                "hide_button": true
            });
            
            // URL 변경 시 즉시 알림 및 3회 루프마다 생존 신호(Heartbeat) 발송
            if active_url != last_detected_url || current_is_shop != last_is_shop {
                let _ = app_handle.emit("browser-match-found", &payload);
                let _ = app_handle.emit("browser-status", &payload);
                
                last_detected_url = active_url;
                last_is_shop = current_is_shop;
                
                if current_is_shop {
                    println!("[AUTO] Active Shop Context Sync: {}", last_detected_url);
                    let tmp_root = crate::utils::paths::get_app_tmp_root(None);
                    let _ = std::fs::create_dir_all(&tmp_root);
                    let shared_data = json!({
                        "origin": last_detected_url,
                        "type": "",
                        "step": "idle",
                        "session_id": "",
                        "kv_path": crate::utils::paths::get_kv_dir(None).to_string_lossy().into_owned()
                    });
                    if let Ok(json_str) = serde_json::to_string(&shared_data) {
                        let _ = std::fs::write(tmp_root.join("index.json"), json_str);
                    }
                }
            } else {
                // 변경이 없더라도 브라우저가 실행 중임을 UI에 주기적으로 알려 플리커링 방지
                let _ = app_handle.emit("browser-status", &payload);
            }
            tokio::time::sleep(Duration::from_millis(800)).await; 
        }
        
        let mut global = GLOBAL_BROWSER.lock().await;
        *global = None;
        
        let _ = app_handle.emit("browser-match-found", json!({ "url": "", "is_client": false, "is_admin": false }));
        println!("[AUTO] Browser monitor exited cleanly.");
    });
}

async fn run_driverless_automation(browser: &str, url: &str, _script: &str, app_handle: tauri::AppHandle) -> anyhow::Result<String> {
    println!("[AUTO] Request: Driverless Automation for {} (URL: {})", browser, url);
    
    // 0. Proactively try to reconnect or reuse if global exists
    let browser_arc = {
        let mut global = GLOBAL_BROWSER.lock().await;
        if global.is_none() {
            // Try to connect to 9222 before launching
            if tokio::net::TcpStream::connect(format!("127.0.0.1:{}", CHROME_DEBUG_PORT)).await.is_ok() {
                println!("[AUTO] Found existing browser on port {}. Connecting...", CHROME_DEBUG_PORT);
                if let Ok((b, mut handler)) = Browser::connect(format!("http://127.0.0.1:{}", CHROME_DEBUG_PORT)).await {
                    let b_arc = Arc::new(b);
                    *global = Some(b_arc.clone());
                    spawn_browser_monitor(b_arc.clone(), app_handle.clone());
                    
                    tokio::spawn(async move {
                        while let Some(h) = handler.next().await { if let Err(_) = h { break; } }
                        let mut g = GLOBAL_BROWSER.lock().await; *g = None; 
                        
                    });
                }
            }
        }
        global.as_ref().cloned()
    };

    let browser_arc = if let Some(b) = browser_arc {
        println!("[AUTO] Reusing/Connected browser instance.");
        let _ = app_handle.emit("browser-status", "running");
        b
    } else {
        // 1. Find Executable Path
        let exec_path = find_browser_path(browser)
            .ok_or_else(|| anyhow!("Browser executable not found for {}", browser))?;
        
        // 2. Build Config
        let build_config = || -> anyhow::Result<BrowserConfig> {
            let port_arg = format!("--remote-debugging-port={}", CHROME_DEBUG_PORT);
            
            let mut args = vec![
                "--start-maximized", 
                "--disable-gpu", 
                "--disable-software-rasterizer",
                "--disable-gpu-compositing",
                "--no-first-run",
                "--disable-notifications",
                "--disable-extensions",
                "--disable-popup-blocking",
                "--blink-settings=imagesEnabled=false",
                "--disable-blink-features=AutomationControlled", // [CRITICAL] Hide automation status
                "--password-store=basic", // Prevent password manager popups
                "--no-default-browser-check",
                "--force-dark-mode", // [THEME] Enable Dark Mode by default
                "--enable-features=WebUIDarkMode", // Ensure UI elements are dark
                &port_arg,
                "--remote-allow-origins=*", 
            ];

            
            let target_url = if url.is_empty() { "about:blank" } else { url };
            args.push(target_url);

            let mut builder = BrowserConfig::builder()
                .chrome_executable(&exec_path)
                .with_head()
                .no_sandbox()
                .viewport(None);

            let tmp_root = crate::utils::paths::get_app_tmp_root(None);
            let profile_dir = tmp_root.join("browser_profiles").join(browser);
            let _ = std::fs::create_dir_all(&profile_dir);
            let mut p_str = std::fs::canonicalize(&profile_dir).unwrap_or(profile_dir).to_string_lossy().to_string();
            if p_str.starts_with(r"\\?\") { p_str = p_str[4..].to_string(); }
            
            builder = builder.user_data_dir(std::path::PathBuf::from(p_str)).args(args);
            builder.build().map_err(|e| anyhow!("Config error: {}", e))
        };

        // 3. Launch
        let (browser_manager, mut handler) = Browser::launch(build_config()?).await
            .map_err(|e| anyhow!("Launch failed: {}", e))?;

        let new_arc = Arc::new(browser_manager);
        {
            let mut global = GLOBAL_BROWSER.lock().await;
            *global = Some(new_arc.clone());
        }
        let _ = app_handle.emit("browser-status", "running");
        spawn_browser_monitor(new_arc.clone(), app_handle.clone());
        tokio::spawn(async move {
            while let Some(h) = handler.next().await { if let Err(_) = h { break; } }
            let mut global = GLOBAL_BROWSER.lock().await; *global = None; 
            
        });
        new_arc
    };

    println!("[AUTO] Targeting initial page for {}...", url);
    
    
    // 🌟 [CRITICAL FIX] 500ms 강제 지연(Sleep)을 2연속 하던 로직을 폐기하고, 
    // 50ms 간격의 초고속 폴링(Fast Polling)으로 교체하여 탭이 생성되는 즉시 반응하도록 최적화합니다.
    let mut pages = vec![];
    for _ in 0..20 {
        if let Ok(p) = browser_arc.pages().await {
            if !p.is_empty() {
                pages = p;
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    
    if pages.is_empty() {
        pages = browser_arc.pages().await.unwrap_or_default();
    }
    
    let nav_target = if url.is_empty() { "about:blank" } else { url };
    
    // 🌟 [CRITICAL FIX] More 버튼 등 특정 URL 요청 시 기존 탭을 무시하고 항상 '새 탭(New Tab)'으로 엽니다.
    // 새 탭을 열면 브라우저가 강제로 화면 맨 앞으로 팝업(Focus)되는 효과도 얻을 수 있어 '반응 없음' 문제가 해결됩니다.
    let page = if nav_target == "about:blank" {
        // 단순 런처 버튼 클릭 시: 기존 활성 탭을 그대로 유지
        if let Some(last_page) = pages.last() {
            last_page.clone()
        } else {
            browser_arc.new_page(nav_target).await.map_err(|e| anyhow!("Page creation failed: {}", e))?
        }
    } else {
        // 특정 URL 이동 요청 시: 무조건 새 탭 생성 (자동 팝업 및 포커스 전환)
        let p = browser_arc.new_page(nav_target).await.map_err(|e| anyhow!("Page creation failed: {}", e))?;
        
        // 만약 방금 켜진 브라우저라서 첫 탭이 빈 화면(newtab) 하나뿐이라면 깔끔하게 닫아줍니다.
        if pages.len() == 1 {
            let first_page = pages.first().unwrap();
            let first_url = first_page.url().await.unwrap_or_default().unwrap_or_default();
            if first_url.is_empty() || first_url == "about:blank" || first_url.contains("newtab") || first_url.contains("new-tab") {
                // 🌟 [CRITICAL FIX] `first_page`는 참조자(&)이므로 소유권(Ownership) 이동을 위해 .clone()을 호출한 뒤 닫습니다.
                let _ = first_page.clone().close().await;
            }
        }
        p
    };

    // [CRITICAL STEALTH] 탐지 우회 스크립트 설정
    let _ = page.evaluate_on_new_document("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})").await;

    Ok(format!("Automation Started."))
}

async fn run_driver_automation(browser: &str, url: &str, script: &str) -> anyhow::Result<String> {
    let (driver_binary, port, capabilities) = match browser {
        "firefox" => {
            let name = if cfg!(target_os = "windows") { "geckodriver.exe" } 
                       else if cfg!(target_os = "macos") { "geckodriver_mac" } 
                       else { "geckodriver" };
             (name.to_string(), DRIVER_PORT, serde_json::json!({ "browserName": "firefox" }))
        },
        "safari" => {
             if cfg!(target_os = "macos") { ("/usr/bin/safaridriver".to_string(), DRIVER_PORT, serde_json::json!({ "browserName": "safari" })) } 
             else { return Err(anyhow!("Safari is only supported on macOS")); }
        },
        _ => return Err(anyhow!("Unsupported browser for driver mode")),
    };

    if !cfg!(target_os = "windows") {
        let _ = Command::new("pkill").arg("-f").arg(&driver_binary).output();
    }

    let mut child = Command::new(&driver_binary)
        .arg(format!("--port={}", port)).stdout(Stdio::null()).stderr(Stdio::null()).spawn()
        .map_err(|e| anyhow!("Driver '{}' start failed: {}", driver_binary, e))?;

    tokio::time::sleep(Duration::from_secs(2)).await;
    let mut caps_map = serde_json::map::Map::new();
    if let Some(obj) = capabilities.as_object() { caps_map.clone_from(obj); }
    
    let client = ClientBuilder::native().capabilities(caps_map).connect(&format!("http://localhost:{}", port)).await;
    if let Err(e) = client { let _ = child.kill(); return Err(anyhow!("Failed to connect to driver: {}", e)); }
    let client = client.unwrap();

    let res = async {
        client.goto(url).await?;
        let result = client.execute(script, vec![]).await?;
        Ok(format!("Driver Success ({}). Result: {}", browser, serde_json::to_string_pretty(&result).unwrap_or_default()))
    }.await;

    let _ = child.kill();
    res
}

pub async fn extract_html_from_current_tab() -> Result<String, String> {
    let target_tab_id = {
        let state = LAST_DETECTED_STATE.lock().await;
        state.tab_id.clone()
    };

    let browser_opt = {
        let global = GLOBAL_BROWSER.lock().await;
        global.as_ref().cloned()
    };
    
    if let Some(browser) = browser_opt {
        let pages = browser.pages().await.map_err(|e| e.to_string())?;
        let mut active_page = None;

        if !target_tab_id.is_empty() {
            for page in pages.iter().rev() {
                let script = "window.__logis_tab_id || ''";
                if let Ok(res) = page.evaluate(script).await {
                    if res.into_value::<String>().unwrap_or_default() == target_tab_id {
                        active_page = Some(page);
                        break;
                    }
                }
            }
        }

        if active_page.is_none() {
            for page in pages.iter().rev() {
                if let Ok(res) = page.evaluate("document.hasFocus()").await {
                    if res.into_value::<bool>().unwrap_or(false) {
                        active_page = Some(page);
                        break;
                    }
                }
            }
        }

        if active_page.is_none() {
            for page in pages.iter().rev() {
                let is_visible = match page.evaluate("document.visibilityState").await {
                    Ok(res) => res.into_value::<String>().unwrap_or_default() == "visible",
                    Err(_) => false,
                };
                if is_visible {
                    active_page = Some(page);
                    break;
                }
            }
        }

        let target_page = active_page.or(pages.last());

        if let Some(page) = target_page {
            let js_script = r#"
                (function() {
                    try {
                        const elements = document.querySelectorAll('*');
                        
                        elements.forEach(el => {
                            const style = window.getComputedStyle(el);
                            if (style.position === 'absolute' || style.position === 'fixed') {
                                const currentStyle = el.getAttribute('style') || '';
                                el.setAttribute('data-logis-original-style', currentStyle);
                                el.setAttribute('style', currentStyle + `; position: ${style.position};`);
                            }
                        });
                        
                        const clone = document.documentElement.cloneNode(true);
                        
                        document.querySelectorAll('iframe').forEach(iframe => {
                            try {
                                if (iframe.contentDocument && iframe.contentDocument.documentElement) {
                                    const iframeClone = iframe.contentDocument.documentElement.cloneNode(true);
                                    clone.appendChild(iframeClone);
                                }
                            } catch(e) {}
                        });

                        document.querySelectorAll('[data-logis-original-style]').forEach(el => {
                            const original = el.getAttribute('data-logis-original-style');
                            if (original) {
                                el.setAttribute('style', original);
                            } else {
                                el.removeAttribute('style');
                            }
                            el.removeAttribute('data-logis-original-style');
                        });
                        
                        clone.querySelectorAll('[data-logis-original-style]').forEach(el => {
                            el.removeAttribute('data-logis-original-style');
                        });
                        
                        return clone.outerHTML;
                    } catch(e) {
                        return document.documentElement.outerHTML;
                    }
                })();
            "#;

            let eval_result = page.evaluate(js_script).await.map_err(|e| e.to_string())?;
            let html = eval_result.into_value::<String>().unwrap_or_default();
            
            return Ok(html);
        }
        Err("No open tabs found.".to_string())
    } else {
        Err("Browser is not running.".to_string())
    }
}

fn is_in_path(cmd: &str) -> bool {
    let check_cmd = if cfg!(target_os = "windows") { "where" } else { "which" };
    Command::new(check_cmd).arg(cmd).stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

fn check_file_exists(path: &str) -> bool { std::path::Path::new(path).exists() }

#[cfg(target_os = "windows")]
fn find_path_in_registry(exe_name: &str) -> Option<String> {
    let queries = [
        format!(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{}", exe_name),
        format!(r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{}", exe_name),
    ];
    for q in queries {
        if let Ok(output) = Command::new("reg").args(["query", &q, "/ve"]).output() {
             if output.status.success() {
                 let s = String::from_utf8_lossy(&output.stdout);
                 if let Some(line) = s.lines().find(|l| l.contains("REG_SZ")) {
                     if let Some(path_part) = line.split("REG_SZ").nth(1) { return Some(path_part.trim().to_string()); }
                 }
             }
        }
    }
    None
}
#[cfg(not(target_os = "windows"))]
fn find_path_in_registry(_: &str) -> Option<String> { None }

#[cfg(target_os = "macos")]
fn find_app_bundle(bundle_id: &str) -> Option<String> {
    let output = Command::new("mdfind").args([format!("kMDItemCFBundleIdentifier == '{}'", bundle_id)]).output();
    if let Ok(o) = output {
        let s = String::from_utf8_lossy(&o.stdout);
        if !s.trim().is_empty() {
             let app_path = s.lines().next().unwrap().trim();
             let binary_name = if bundle_id.contains("Chrome") { "Google Chrome" } else if bundle_id.contains("Edge") { "Microsoft Edge" } else if bundle_id.contains("Firefox") { "firefox" } else { return None };
             if binary_name == "firefox" { return Some(format!("{}/Contents/MacOS/firefox", app_path)); }
             return Some(format!("{}/Contents/MacOS/{}", app_path, binary_name));
        }
    }
    None
}
#[cfg(not(target_os = "macos"))]
fn find_app_bundle(_: &str) -> Option<String> { None }

fn find_browser_path(browser: &str) -> Option<PathBuf> {
    let mut potential_paths = Vec::new();
    match browser {
        "chrome" => {
            if cfg!(target_os = "windows") {
                potential_paths.push(r"C:\Program Files\Google\Chrome\Application\chrome.exe".to_string());
                potential_paths.push(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe".to_string());
                if let Some(p) = find_path_in_registry("chrome.exe") { potential_paths.push(p); }
            } else if cfg!(target_os = "macos") {
                potential_paths.push("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".to_string());
                if let Some(p) = find_app_bundle("com.google.Chrome") { potential_paths.push(p); }
            } else {
                if let Ok(p) = which::which("google-chrome") { return Some(p); }
                if let Ok(p) = which::which("google-chrome-stable") { return Some(p); }
                if let Ok(p) = which::which("chromium") { return Some(p); }
                if let Ok(p) = which::which("chrome") { return Some(p); }
            }
        },
        "edge" => {
            if cfg!(target_os = "windows") {
                potential_paths.push(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe".to_string());
                if let Some(p) = find_path_in_registry("msedge.exe") { potential_paths.push(p); }
            } else if cfg!(target_os = "macos") {
                potential_paths.push("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge".to_string());
                if let Some(p) = find_app_bundle("com.microsoft.edgemac") { potential_paths.push(p); }
            } else {
                if let Ok(p) = which::which("microsoft-edge") { return Some(p); }
                if let Ok(p) = which::which("edge") { return Some(p); }
            }
        },
        "firefox" => {
             if cfg!(target_os = "windows") {
                 potential_paths.push(r"C:\Program Files\Mozilla Firefox\firefox.exe".to_string());
                 potential_paths.push(r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe".to_string());
             } else if cfg!(target_os = "macos") {
                 potential_paths.push("/Applications/Firefox.app/Contents/MacOS/firefox".to_string());
                 if let Some(p) = find_app_bundle("org.mozilla.firefox") { potential_paths.push(p); }
             } else { if let Ok(p) = which::which("firefox") { return Some(p); } }
        },
        _ => return None,
    };
    for p in potential_paths {
        let path = PathBuf::from(&p);
        if path.exists() { return Some(path); }
    }
    match browser {
        "chrome" => which::which("chrome").ok(),
        "edge" => which::which("msedge").ok(),
        "firefox" => which::which("firefox").ok(),
        _ => None,
    }
}

pub fn get_available_browsers() -> Vec<BrowserStatus> {
    let mut browsers = Vec::new();
    if find_browser_path("chrome").is_some() {
        browsers.push(BrowserStatus { name: "chrome".to_string(), is_supported: true, is_installed: true, needs_driver: false });
    }
    if find_browser_path("edge").is_some() {
         browsers.push(BrowserStatus { name: "edge".to_string(), is_supported: true, is_installed: true, needs_driver: false });
    }
    if find_browser_path("firefox").is_some() {
        let driver_name = if cfg!(target_os = "windows") { "geckodriver.exe" } else { "geckodriver" };
        let has_driver = is_in_path(driver_name) || check_file_exists(driver_name);
        browsers.push(BrowserStatus { name: "firefox".to_string(), is_supported: true, is_installed: true, needs_driver: !has_driver });
    }
    if cfg!(target_os = "macos") {
        browsers.push(BrowserStatus { name: "safari".to_string(), is_supported: true, is_installed: true, needs_driver: true });
    }
    browsers
}