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
                let _ = app_handle.emit("browser-status", "stopped");
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

fn spawn_browser_monitor(browser: Arc<Browser>, app_handle: tauri::AppHandle) {
    tokio::spawn(async move {
        let mut last_detected_url = String::new();
        loop {
            let pages = match browser.pages().await {
                Ok(p) => p,
                Err(_) => break, 
            };

            let mut active_page = None;
            for page in pages.iter() {
                let is_visible = match page.evaluate("document.visibilityState").await {
                    Ok(res) => res.into_value::<String>().unwrap_or_default() == "visible",
                    Err(_) => false,
                };
                if is_visible {
                    active_page = Some(page);
                    break;
                }
            }

            // Fallback: If no tab is strictly 'visible', use the last one in the list
            let page_to_check = active_page.or(pages.last());

            if let Some(page) = page_to_check {
                if let Ok(val) = page.evaluate("window.location.href").await {
                    let current_url = val.into_value::<String>().unwrap_or_default();
                    if !current_url.is_empty() && current_url != last_detected_url {
                        let is_client = is_shop(&current_url, CLIENT_PATTERNS);
                        let is_admin = is_shop(&current_url, ADMIN_PATTERNS);
                        
                        // Always notify frontend of URL change so it can toggle button visibility
                        let payload = json!({
                            "url": current_url.clone(),
                            "is_client": is_client,
                            "is_admin": is_admin
                        });
                        let _ = app_handle.emit("browser-match-found", &payload);
                        
                        if is_client || is_admin {
                            println!("[AUTO] Target Site Detected: {} (Client={}, Admin={})", current_url, is_client, is_admin);
                        }
                        last_detected_url = current_url;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(1000)).await; 
        }
        // When loop breaks (browser closed), clear detected URL in frontend
        let _ = app_handle.emit("browser-match-found", json!({ "url": "", "is_client": false, "is_admin": false }));
        println!("[AUTO] Browser monitor exited.");
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
                    
                    let app_handle_clone = app_handle.clone();
                    tokio::spawn(async move {
                        while let Some(h) = handler.next().await { if let Err(_) = h { break; } }
                        let mut g = GLOBAL_BROWSER.lock().await; *g = None; 
                        let _ = app_handle_clone.emit("browser-status", "stopped");
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
            let args = vec![
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
                &port_arg,
                "--remote-allow-origins=*", 
            ];
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
            builder = builder.user_data_dir(std::path::PathBuf::from(p_str));
            builder.args(args).build().map_err(|e| anyhow!("Config error: {}", e))
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
        let app_handle_clone = app_handle.clone();
        tokio::spawn(async move {
            while let Some(h) = handler.next().await { if let Err(_) = h { break; } }
            let mut global = GLOBAL_BROWSER.lock().await; *global = None; 
            let _ = app_handle_clone.emit("browser-status", "stopped");
        });
        new_arc
    };

    println!("[AUTO] Navigating to {}...", url);
    let page = browser_arc.new_page(url).await.map_err(|e| anyhow!("Page creation failed: {}", e))?;
    
    // [CRITICAL STEALTH] Redefine navigator.webdriver to bypass detection
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
    let browser_opt = {
        let global = GLOBAL_BROWSER.lock().await;
        global.as_ref().cloned()
    };
    if let Some(browser) = browser_opt {
        let pages = browser.pages().await.map_err(|e| e.to_string())?;
        if let Some(page) = pages.last() {
            let html = page.content().await.map_err(|e| e.to_string())?;
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