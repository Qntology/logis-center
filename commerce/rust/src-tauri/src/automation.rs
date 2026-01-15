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
use serde_json::{Value, json};

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
        // Notify Frontend: Running
        let _ = app_handle.emit("browser-status", "running");
        b
    } else {
        // 1. Find Executable Path
        let exec_path = find_browser_path(browser)
            .ok_or_else(|| anyhow!("Browser executable not found for {}", browser))?;
        
        println!("[AUTO] Using executable: {:?}", exec_path);

        // Get profile root if exists
        let profile_root = find_profile_root(browser);

        // Function to build config
        let build_config = |_: bool| -> anyhow::Result<BrowserConfig> {
            let mut args = vec![
                "--start-maximized", 
                "--disable-gpu", 
                "--disable-infobars",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-session-crashed-bubble",
                "--disable-features=Translate",
                "--remote-debugging-port=0"
            ];

            let mut builder = BrowserConfig::builder()
                .chrome_executable(&exec_path) 
                .with_head() 
                .no_sandbox()
                .viewport(None);
            
            // Use App-Specific Profile for persistence without conflicts
            // We use a fixed relative path "browser_profile" in the current working directory
            // or we could use app_handle.path().app_data_dir() if we want it hidden.
            // For portability, let's use "./browser_profile/{browser}"
            let profile_dir = std::path::Path::new("browser_profiles").join(browser);
            let _ = std::fs::create_dir_all(&profile_dir);
            
            let abs_profile_dir = std::fs::canonicalize(&profile_dir).unwrap_or(profile_dir);
            
            println!("[AUTO] Using isolated app profile at: {:?}", abs_profile_dir);
            builder = builder.user_data_dir(abs_profile_dir);

            builder.args(args).build().map_err(|e| anyhow!("Failed to build config: {}", e))
        };

        // 2. Launch Strategy (Simplified: Single Attempt with Long Timeout)
        let launch_result: anyhow::Result<(Browser, Handler)> = async {
            // We always use the custom profile now, so 'true'/'false' flag is irrelevant
            let config = build_config(true)?; 
            println!("[AUTO] Launching browser... (Timeout: 60s)");
            
            let launch_attempt = tokio::time::timeout(
                Duration::from_secs(60), 
                Browser::launch(config)
            ).await;

            match launch_attempt {
                Ok(Ok(res)) => Ok(res),
                Ok(Err(e)) => Err(anyhow!("Launch Error: {}", e)),
                Err(_) => Err(anyhow!("Launch Timeout (60s). Chrome opened but didn't connect. Please check if a dialog is blocking it.")),
            }
        }.await;

        let (browser_manager, mut handler) = launch_result?;
        let new_arc = Arc::new(browser_manager);

        // Store Globally
        {
            let mut global = GLOBAL_BROWSER.lock().await;
            *global = Some(new_arc.clone());
        }
        
        // Notify Frontend: Running
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
            // Clear global on exit
            println!("[AUTO] Browser Handler Exited. Cleaning up global instance...");
            let mut global = GLOBAL_BROWSER.lock().await; 
            *global = None; 
            
            // Notify Frontend: Stopped
            let _ = app_handle_clone.emit("browser-status", "stopped");
        });

        new_arc
    };

    println!("[AUTO] Setting up page...");
    
    // 3. Create or Get Page
    let mut page = browser_arc.new_page("about:blank").await
        .map_err(|e| anyhow!("Failed to create page: {}", e))?;

    // 4. Inject Script on EVERY Navigation
    println!("[AUTO] Injecting persistent script...");

    // --- Bundle Scripts (Pako -> Content.js) ---
    // Using include_str! to embed assets at compile time
    let pako_lib = include_str!("assets/scripts/pako.min.js");
    // Dexie removed as we now use Rust LanceDB
    let content_logic = include_str!("assets/scripts/content.js");

    // If a custom script is provided, append it; otherwise use the default bundle
    let final_script = if script.is_empty() {
        format!("{};\n{};", pako_lib, content_logic)
    } else {
        format!("{};\n{};\n{}", pako_lib, content_logic, script)
    };
    
use tauri::{Emitter, Manager}; 
use crate::AppState; 
use crate::store::VectorStore;
use futures::future::BoxFuture; // Added import

// ...

    // --- ADDED: JS to Rust Binding ---
    // This allows content.js to call window.__TAURI_POST_TASK__(data)
    let app_handle_for_event = app_handle.clone();
    
    let callback = move |mut args: Vec<serde_json::Value>| -> BoxFuture<'static, anyhow::Result<serde_json::Value>> {
        let app_handle = app_handle_for_event.clone();
        Box::pin(async move {
            let payload = args.pop().unwrap_or(json!({}));
            
            // ... (rest of the logic identical to before)
            // Check for DB commands (select, upsert, delete, clear)
            let is_db_cmd = payload.get("select").is_some() || 
                           payload.get("upsert").is_some() || 
                           payload.get("delete").is_some() ||
                           payload.get("clear").is_some();

            if is_db_cmd {
                let state = app_handle.state::<AppState>();
                let store_guard = state.store.lock().await;
                
                if let Some(store) = store_guard.as_ref() {
                    if let Some(table) = payload.get("select").and_then(|s| s.as_str()) {
                        let db_table = match table {
                            "items" => "commerce_items",
                            "pages" => "commerce_items", 
                            "users" => "commerce_users",
                            "crons" => "tasks",
                            _ => table
                        };
                        
                        if let (Some(key), Some(value)) = (payload.get("key").and_then(|s| s.as_str()), payload.get("value")) {
                            if let Ok(Some((id, data))) = store.find_item_by_property(db_table, key, value).await {
                                let mut res = data.clone();
                                if let Some(obj) = res.as_object_mut() {
                                    obj.insert("id".to_string(), json!(id));
                                }
                                return Ok(json!({ "results": [res] }));
                            } else {
                                return Ok(json!({ "results": [] }));
                            }
                        } else {
                             return Ok(json!({ "results": [] }));
                        }
                    } 
                    else if let Some(table) = payload.get("upsert").and_then(|s| s.as_str()) {
                         let db_table = match table {
                            "items" => "commerce_items",
                            "pages" => "commerce_items",
                            "users" => "commerce_users",
                            "crons" => "tasks",
                            _ => table
                        };
                        
                        if let Some(val) = payload.get("value") {
                            let id = val.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                            let type_ = val.get("type").and_then(|s| s.as_str()).unwrap_or(table);
                            
                            if db_table == "tasks" {
                                let task = crate::store::Task {
                                    id: id.clone(),
                                    r#type: val.get("job").and_then(|s| s.as_str()).unwrap_or("cron").to_string(),
                                    from_source: "browser".to_string(),
                                    to_dest: "local".to_string(),
                                    cc: val.get("cc").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                                    bcc: val.get("bcc").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                                    ref_id: val.get("ref").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                                    data_json: val.to_string(),
                                    created_at: val.get("created_at").and_then(|v| v.as_i64()).unwrap_or(0),
                                    updated_at: val.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0),
                                    status: "pending".to_string(),
                                };
                                let _ = store.add_task(task).await;
                                return Ok(json!({ "results": [val] }));
                            }
                            
                            let _ = store.upsert_item(db_table, &id, type_, val.clone(), None).await;
                            return Ok(json!({ "results": [val] }));
                        }
                    }
                    else if let Some(table) = payload.get("delete").and_then(|s| s.as_str()) {
                        return Ok(json!({ "results": [] }));
                    }
                }
                
                return Ok(json!({ "results": [], "error": "Store not initialized" }));
            }

            println!("[AUTO] Event received from JS. Emitting 'new-task-from-browser'...");
            app_handle.emit("new-task-from-browser", &payload).unwrap();
            
            Ok(json!({ "status": "event_emitted" }))
        })
    };

    /*
    page.expose_function("__TAURI_POST_TASK__", callback).await.map_err(|e| anyhow!("Failed to expose function: {}", e))?;
    */

    page.evaluate_on_new_document(&final_script).await
        .map_err(|e| anyhow!("Failed to set up script injection: {}", e))?;

    // 5. Navigate to Target URL
    println!("[AUTO] Navigating to {}", url);
    page.goto(url).await
        .map_err(|e| anyhow!("Navigation failed: {}", e))?;

    Ok(format!("Automation Started. Script will run on every page load in this tab."))
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
                "geckodriver" 
            };
             (name.to_string(), DRIVER_PORT, serde_json::json!({ "browserName": "firefox" }))
        },
        "safari" => {
             ("/usr/bin/safaridriver".to_string(), DRIVER_PORT, serde_json::json!({ "browserName": "safari" }))
        },
        _ => return Err(anyhow!("Unsupported browser for driver mode")),
    };

    println!("[AUTO] Legacy Mode: {} requires driver {}", browser, driver_binary);
    
    let mut child = Command::new(&driver_binary)
        .arg(format!("--port={}", port))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_e| anyhow!("Driver '{}' not found. This browser requires a driver.", driver_binary))?;

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

    client.goto(url).await?;
    let result = client.execute(script, vec![]).await?;
    let result_str = serde_json::to_string_pretty(&result).unwrap_or_default();

    Ok(format!("Driver Success ({}). Result: {}", browser, result_str))
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
             let binary_name = if bundle_id.contains("Chrome") { "Google Chrome" } else { "Microsoft Edge" };
             return Some(format!("{}/Contents/MacOS/{}", app_path, binary_name));
        }
    }
    None
}
#[cfg(not(target_os = "macos"))]
fn find_app_bundle(_: &str) -> Option<String> { None }

fn find_profile_root(browser: &str) -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    let path_str = match (cfg!(target_os = "windows"), cfg!(target_os = "macos"), browser) {
        (true, _, "chrome") => format!(r"{}\AppData\Local\Google\Chrome\User Data", home),
        (true, _, "edge") => format!(r"{}\AppData\Local\Microsoft\Edge\User Data", home),
        (_, true, "chrome") => format!(r"{}/Library/Application Support/Google/Chrome", home),
        (_, true, "edge") => format!(r"{}/Library/Application Support/Microsoft Edge", home),
        _ => match browser {
            "chrome" => format!(r"{}/.config/google-chrome", home),
            "edge" => format!(r"{}/.config/microsoft-edge", home),
            _ => return None,
        }
    };

    let path = PathBuf::from(path_str);
    if path.exists() { Some(path) } else { None }
}

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

fn find_browser_path(browser: &str) -> Option<PathBuf> {
    let potential_paths = match browser {
        "chrome" => {
            let mut paths = Vec::new();
            if cfg!(target_os = "windows") {
                paths.push(r"C:\Program Files\Google\Chrome\Application\chrome.exe".to_string());
                paths.push(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe".to_string());
                if let Some(p) = find_path_in_registry("chrome.exe") { paths.push(p); }
            } else if cfg!(target_os = "macos") {
                paths.push("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".to_string());
                if let Some(p) = find_app_bundle("com.google.Chrome") { paths.push(p); }
            } else {
                return which::which("google-chrome").ok()
                    .or_else(|| which::which("google-chrome-stable").ok())
                    .or_else(|| which::which("chromium").ok())
                    .or_else(|| which::which("chrome").ok());
            }
            paths
        },
        "edge" => {
            let mut paths = Vec::new();
            if cfg!(target_os = "windows") {
                paths.push(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe".to_string());
                paths.push(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe".to_string());
                if let Some(p) = find_path_in_registry("msedge.exe") { paths.push(p); }
            } else if cfg!(target_os = "macos") {
                paths.push("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge".to_string());
                if let Some(p) = find_app_bundle("com.microsoft.edgemac") { paths.push(p); }
            } else {
                return which::which("microsoft-edge").ok()
                    .or_else(|| which::which("microsoft-edge-stable").ok())
                    .or_else(|| which::which("edge").ok());
            }
            paths
        },
        "firefox" => {
             if cfg!(target_os = "windows") {
                 vec![
                     r"C:\Program Files\Mozilla Firefox\firefox.exe".to_string(),
                     r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe".to_string(),
                 ]
             } else {
                 return which::which("firefox").ok();
             }
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
        _ => None,
    }
}

pub fn get_available_browsers() -> Vec<BrowserStatus> {
    let mut browsers = Vec::new();
    if find_browser_path("chrome").is_some() {
        browsers.push(BrowserStatus {
            name: "chrome".to_string(),
            is_supported: true,
            is_installed: true,
            needs_driver: false,
        });
    }
    if find_browser_path("edge").is_some() {
         browsers.push(BrowserStatus {
            name: "edge".to_string(),
            is_supported: true,
            is_installed: true,
            needs_driver: false,
        });
    }
    if find_browser_path("firefox").is_some() {
        let driver = if cfg!(target_os = "windows") { 
            "geckodriver.exe" 
        } else if cfg!(target_os = "macos") {
            "geckodriver_mac"
        } else { 
            "geckodriver" 
        };
        let has_driver = is_in_path(driver) || check_file_exists(driver);
        browsers.push(BrowserStatus {
            name: "firefox".to_string(),
            is_supported: true,
            is_installed: true,
            needs_driver: !has_driver,
        });
    }
    if cfg!(target_os = "macos") {
        browsers.push(BrowserStatus {
            name: "safari".to_string(),
            is_supported: true,
            is_installed: true,
            needs_driver: true,
        });
    }
    browsers
}
