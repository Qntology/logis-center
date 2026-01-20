use anyhow::anyhow;
use chromiumoxide::{Browser, BrowserConfig, Handler};
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
static GLOBAL_BROWSER: Lazy<Arc<tokio::sync::Mutex<Option<Arc<Browser>>>>> = Lazy::new(|| {
    Arc::new(tokio::sync::Mutex::new(None))
});

// Driver Port (Only for Firefox/Safari)
const DRIVER_PORT: u16 = 4444;

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
    "partner.wemakeprice.com", "activeitzone.com", "demofran.com",
    // Add exact matches for root domains often used with wildcards
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
    if let Ok(parsed_url) = Url::parse(url) {
        let host = parsed_url.host_str().unwrap_or("");
        for pattern in patterns {
            let regex_str = format!("^{}$", pattern.replace(".", "\\.").replace("*", ".*"));
            if let Ok(re) = Regex::new(&regex_str) {
                if re.is_match(host) {
                    return true;
                }
            }
        }
    } else {
        // Fallback: If URL parsing fails, check if pattern exists in the string directly
        for pattern in patterns {
            let clean_pattern = pattern.replace("*.", "");
            if url.contains(&clean_pattern) {
                return true;
            }
        }
    }
    false
}

// Removed #[tauri::command] to avoid conflict with lib.rs
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

// --- Driverless Automation (Chrome/Edge) ---
async fn run_driverless_automation(browser: &str, url: &str, script: &str, app_handle: tauri::AppHandle) -> anyhow::Result<String> {
    println!("[AUTO] Starting Driverless Automation for {}", browser);

    // 0. Check for existing global browser instance
    let existing_browser = {
        let global = GLOBAL_BROWSER.lock().await;
        global.as_ref().cloned()
    };

    let browser_arc = if let Some(b) = existing_browser {
        println!("[AUTO] Reusing existing browser instance.");
        let _ = app_handle.emit("browser-status", "running");
        b
    } else {
        // 1. Find Executable Path
        let exec_path = find_browser_path(browser)
            .ok_or_else(|| anyhow!("Browser executable not found for {}", browser))?;
        
        println!("[AUTO] Using executable: {:?}", exec_path);

        // Function to build config
        let build_config = |_: bool| -> anyhow::Result<BrowserConfig> {
            let args = vec![
                "--start-maximized", 
                "--disable-gpu", 
                "--disable-infobars",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-session-crashed-bubble",
                "--disable-features=Translate",
                "--remote-debugging-port=0",
                "--remote-allow-origins=*",
                "--disable-extensions"
            ];

            let mut builder = BrowserConfig::builder()
                .chrome_executable(&exec_path) 
                .with_head() 
                .no_sandbox()
                .viewport(None);
            
            let profile_dir = std::path::Path::new("browser_profiles").join(browser);
            let _ = std::fs::create_dir_all(&profile_dir);
            let abs_profile_dir = std::fs::canonicalize(&profile_dir).unwrap_or(profile_dir);
            
            let mut profile_path_str = abs_profile_dir.to_string_lossy().to_string();
            // [FIX] Strip UNC prefix (\\?\) on Windows which causes Chrome to fail
            if profile_path_str.starts_with(r"\\?\") {
                profile_path_str = profile_path_str[4..].to_string();
            }
            
            println!("[AUTO] Using isolated app profile at: {}", profile_path_str);
            builder = builder.user_data_dir(std::path::PathBuf::from(profile_path_str));

            builder.args(args).build().map_err(|e| anyhow!("Failed to build config: {}", e))
        };

        // 2. Launch Strategy
        let launch_result: anyhow::Result<(Browser, Handler)> = async {
            let config = build_config(true)?;
            println!("[AUTO] Launching browser... (Timeout: 60s)");
            
            let launch_attempt = tokio::time::timeout(
                Duration::from_secs(60), 
                Browser::launch(config)
            ).await;

            match launch_attempt {
                Ok(Ok(res)) => Ok(res),
                Ok(Err(e)) => Err(anyhow!("Launch Error: {}", e)),
                Err(_) => Err(anyhow!("Launch Timeout. Check if a dialog is blocking it.")),
            }
        }.await;

        let (browser_manager, mut handler) = launch_result?;
        let new_arc = Arc::new(browser_manager);

        // Store Globally
        {
            let mut global = GLOBAL_BROWSER.lock().await;
            *global = Some(new_arc.clone());
        }
        
        let _ = app_handle.emit("browser-status", "running");

        // Spawn Handler
        let app_handle_clone = app_handle.clone();
        tokio::spawn(async move {
            while let Some(h) = handler.next().await {
                if let Err(e) = h {
                    println!("[AUTO-DEBUG] Browser Handler Event: {:?}", e);
                    break;
                }
            }
            println!("[AUTO] Browser Handler Exited.");
            let mut global = GLOBAL_BROWSER.lock().await; 
            *global = None; 
            let _ = app_handle_clone.emit("browser-status", "stopped");
        });

        new_arc
    };

    println!("[AUTO] Setting up page...");
    
    // 3. Create Page & Navigate
    let page = browser_arc.new_page(url).await
        .map_err(|e| anyhow!("Failed to create page: {}", e))?;

    // 4. Inject Persistent Script (Pako + Content.js)
    let pako_lib = include_str!("assets/scripts/pako.min.js");
    let content_logic = include_str!("assets/scripts/content.js");
    let final_script = if script.is_empty() {
        format!("{};\n{};", pako_lib, content_logic)
    } else {
        format!("{};\n{};\n{}", pako_lib, content_logic, script)
    };

    // Inject on new document creation (navigation)
    let _ = page.evaluate(final_script).await;


    // 5. Start Background Monitoring Loop
    // Improved loop to detect the ACTUAL active/visible tab
    let monitor_browser = browser_arc.clone();
    let monitor_handle = app_handle.clone();
    
    tokio::spawn(async move {
        let mut last_detected_url = String::new();

        loop {
            let pages = match monitor_browser.pages().await {
                Ok(p) => p,
                Err(_) => break, 
            };

            let mut _active_tab_found = false;

            for (_i, page) in pages.iter().enumerate() {
                // Check if this tab is the active one (visible)
                let is_visible = match page.evaluate("document.visibilityState").await {
                    Ok(res) => {
                        res.into_value::<String>().unwrap_or_default() == "visible"
                    },
                    Err(_) => false,
                };
                
                if is_visible {
                     // println!("[AUTO-DEBUG] Found visible tab: {:?}", page.url().await);
                } else {
                     // println!("[AUTO-DEBUG] Tab {} is hidden", _i);
                }

                if is_visible {
                    _active_tab_found = true;
                    
                    // Use JS evaluation for reliable URL detection (fixes bookmark/stale state issues)
                    let current_url_res = page.evaluate("window.location.href").await;
                    
                    if let Ok(val) = current_url_res {
                        let current_url = val.into_value::<String>().unwrap_or_default();
                        
                        // Only emit if the URL implies a change in state or it's a new URL
                        if current_url != last_detected_url {
                            let is_client = is_shop(&current_url, CLIENT_PATTERNS);
                            let is_admin = is_shop(&current_url, ADMIN_PATTERNS);
                            
                            let _ = monitor_handle.emit("browser-match-found", json!({
                                "url": current_url.clone(),
                                "is_client": is_client,
                                "is_admin": is_admin
                            }));
                            last_detected_url = current_url;
                        }
                    }
                    // Found the active tab, no need to check others
                    break;
                }
            }

            // Optional: If no tab is visible (e.g. browser minimized or separate issue), 
            // we might want to retain the last state or do nothing. 
            // Currently, we just wait for the next poll.

            tokio::time::sleep(Duration::from_millis(500)).await; 
        }
    });

    Ok(format!("Automation Started. Monitoring for target sites..."))
}

// --- HTML Extraction Command ---
// Removed #[tauri::command] to avoid conflict
pub async fn extract_html_from_current_tab() -> Result<String, String> {
    let browser_opt = {
        let global = GLOBAL_BROWSER.lock().await;
        global.as_ref().cloned()
    };

    if let Some(browser) = browser_opt {
        let pages = browser.pages().await.map_err(|e| e.to_string())?;
        // Use the last page as the active one
        if let Some(page) = pages.last() {
            // content() returns the full HTML of the page
            let html = page.content().await.map_err(|e| e.to_string())?;
            return Ok(html);
        }
        Err("No open tabs found.".to_string())
    } else {
        Err("Browser is not running.".to_string())
    }
}


// --- Legacy Driver Automation (Firefox/Safari) ---
async fn run_driver_automation(browser: &str, url: &str, script: &str) -> anyhow::Result<String> {
    let (driver_binary, port, capabilities) = match browser {
        "firefox" => {
            let name = if cfg!(target_os = "windows") { 
                "geckodriver.exe" 
            } else if cfg!(target_os = "macos") {
                "geckodriver_mac"
            } else { 
                "geckodriver" // Linux & others
            };
             (name.to_string(), DRIVER_PORT, serde_json::json!({ "browserName": "firefox" }))
        },
        "safari" => {
             if cfg!(target_os = "macos") {
                 ("/usr/bin/safaridriver".to_string(), DRIVER_PORT, serde_json::json!({ "browserName": "safari" }))
             } else {
                 return Err(anyhow!("Safari is only supported on macOS"));
             }
        },
        _ => return Err(anyhow!("Unsupported browser for driver mode")),
    };

    println!("[AUTO] Legacy Mode: {} requires driver {}", browser, driver_binary);
    
    // Attempt to kill existing driver instances on port (Linux/Mac mostly)
    if !cfg!(target_os = "windows") {
        let _ = Command::new("pkill").arg("-f").arg(&driver_binary).output();
    }

    let mut child = Command::new(&driver_binary)
        .arg(format!("--port={}", port))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| anyhow!("Driver '{}' start failed. Ensure it is installed. Error: {}", driver_binary, e))?;

    // Give driver time to start
    tokio::time::sleep(Duration::from_secs(2)).await;

    let mut caps_map = serde_json::map::Map::new();
    if let Some(obj) = capabilities.as_object() {
        caps_map.clone_from(obj);
    }
    
    let client = ClientBuilder::native()
        .capabilities(caps_map)
        .connect(&format!("http://localhost:{}", port))
        .await;
        
    if let Err(e) = client {
        let _ = child.kill();
        return Err(anyhow!("Failed to connect to driver: {}", e));
    }
    let client = client.unwrap();

    let res = async {
        client.goto(url).await?;
        let result = client.execute(script, vec![]).await?;
        let result_str = serde_json::to_string_pretty(&result).unwrap_or_default();
        Ok(format!("Driver Success ({}). Result: {}", browser, result_str))
    }.await;

    let _ = child.kill(); // Ensure driver is killed after execution
    res
}

// --- Helper Functions ---

fn is_in_path(cmd: &str) -> bool {
    let check_cmd = if cfg!(target_os = "windows") { "where" } else { "which" };
    Command::new(check_cmd).arg(cmd).stdout(Stdio::null()).stderr(Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false)
}

fn check_file_exists(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

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
                     if let Some(path_part) = line.split("REG_SZ").nth(1) {
                         return Some(path_part.trim().to_string());
                     }
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
             let app_path = s.lines().next()?.trim();
             let binary_name = if bundle_id.contains("Chrome") { "Google Chrome" } 
                               else if bundle_id.contains("Edge") { "Microsoft Edge" }
                               else if bundle_id.contains("Firefox") { "firefox" }
                               else { return None };
             
             // Firefox bundle binary location is slightly different often
             if binary_name == "firefox" {
                 return Some(format!("{}/Contents/MacOS/firefox", app_path));
             }
             return Some(format!("{}/Contents/MacOS/{}", app_path, binary_name));
        }
    }
    None
}
#[cfg(not(target_os = "macos"))]
fn find_app_bundle(_: &str) -> Option<String> { None }

#[allow(dead_code)]
fn find_profile_root(browser: &str) -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    let path_str = match (cfg!(target_os = "windows"), cfg!(target_os = "macos"), browser) {
        (true, _, "chrome") => format!(r"{}\AppData\Local\Google\Chrome\User Data", home),
        (true, _, "edge") => format!(r"{}\AppData\Local\Microsoft\Edge\User Data", home),
        (_, true, "chrome") => format!("{}/Library/Application Support/Google/Chrome", home),
        (_, true, "edge") => format!("{}/Library/Application Support/Microsoft Edge", home),
        _ => match browser {
            "chrome" => format!("{}/.config/google-chrome", home),
            "edge" => format!("{}/.config/microsoft-edge", home),
            _ => return None,
        }
    };

    let path = PathBuf::from(path_str);
    if path.exists() { Some(path) } else { None }
}

#[allow(dead_code)]
fn get_first_profile_name(root: &PathBuf) -> String {
    if root.join("Default").exists() {
        return "Default".to_string();
    }
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut profiles: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|name| name.starts_with("Profile "))
            .collect();
        profiles.sort();
        if let Some(first) = profiles.first() {
            return first.clone();
        }
    }
    "Default".to_string()
}

// Find browser binary path across Windows, Mac, and Linux
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
                // Linux / Unix
                if let Ok(p) = which::which("google-chrome") { return Some(p); }
                if let Ok(p) = which::which("google-chrome-stable") { return Some(p); }
                if let Ok(p) = which::which("chromium") { return Some(p); }
                if let Ok(p) = which::which("chromium-browser") { return Some(p); }
                if let Ok(p) = which::which("chrome") { return Some(p); }
            }
        },
        "edge" => {
            if cfg!(target_os = "windows") {
                potential_paths.push(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe".to_string());
                potential_paths.push(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe".to_string());
                if let Some(p) = find_path_in_registry("msedge.exe") { potential_paths.push(p); }
            } else if cfg!(target_os = "macos") {
                potential_paths.push("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge".to_string());
                if let Some(p) = find_app_bundle("com.microsoft.edgemac") { potential_paths.push(p); }
            } else {
                // Linux / Unix
                if let Ok(p) = which::which("microsoft-edge") { return Some(p); }
                if let Ok(p) = which::which("microsoft-edge-stable") { return Some(p); }
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
             } else {
                 // Linux / Unix
                 if let Ok(p) = which::which("firefox") { return Some(p); }
             }
        },
        _ => return None,
    };

    // Check hardcoded/detected paths
    for p in potential_paths {
        let path = PathBuf::from(&p);
        if path.exists() { return Some(path); }
    }

    // Fallback: simple command check
    match browser {
        "chrome" => which::which("chrome").ok(),
        "edge" => which::which("msedge").ok(),
        "firefox" => which::which("firefox").ok(),
        _ => None,
    }
}

pub fn get_available_browsers() -> Vec<BrowserStatus> {
    let mut browsers = Vec::new();
    
    // Check Chrome
    if find_browser_path("chrome").is_some() {
        browsers.push(BrowserStatus {
            name: "chrome".to_string(),
            is_supported: true,
            is_installed: true,
            needs_driver: false,
        });
    }
    
    // Check Edge
    if find_browser_path("edge").is_some() {
         browsers.push(BrowserStatus {
            name: "edge".to_string(),
            is_supported: true,
            is_installed: true,
            needs_driver: false,
        });
    }
    
    // Check Firefox
    if find_browser_path("firefox").is_some() {
        let driver_name = if cfg!(target_os = "windows") { "geckodriver.exe" } else { "geckodriver" };
        let has_driver = is_in_path(driver_name) || check_file_exists(driver_name);
        browsers.push(BrowserStatus {
            name: "firefox".to_string(),
            is_supported: true,
            is_installed: true,
            needs_driver: !has_driver,
        });
    }
    
    // Check Safari (Mac Only)
    if cfg!(target_os = "macos") {
        browsers.push(BrowserStatus {
            name: "safari".to_string(),
            is_supported: true,
            is_installed: true,
            needs_driver: true, // Always needs driver mode
        });
    }
    
    browsers
}
