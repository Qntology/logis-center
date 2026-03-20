use rcgen::{CertificateParams, KeyPair, PKCS_ECDSA_P256_SHA256, DistinguishedName};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use once_cell::sync::Lazy;

// [NEW] 대기 중인 응답 소켓들을 관리하기 위한 글로벌 맵
static PENDING_ANSWERS: Lazy<Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(HashMap::new()))
});

#[derive(Serialize, Deserialize, Debug)]
pub struct SignalMessage {
    pub seed: u64,
    pub sdp: String,
}

mod commands {
    use super::*;
    use tauri::command;

    #[command]
    pub fn get_deterministic_cert(seed_num: u64) -> (String, String) {
        let _rng = ChaCha20Rng::seed_from_u64(seed_num);
        let key_pair = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut params = CertificateParams::new(vec!["LocalNode".to_string()]).unwrap();
        params.distinguished_name = DistinguishedName::new();
        params.distinguished_name.push(rcgen::DnType::CommonName, "LocalNode");
        let cert = params.self_signed(&key_pair).unwrap();
        (cert.pem(), key_pair.serialize_pem())
    }

    #[command]
    pub fn get_mobile_ip() -> String {
        let socket = match std::net::UdpSocket::bind("0.0.0.0:0") {
            Ok(s) => s,
            Err(_) => return "127.0.0.1".to_string(),
        };
        if socket.connect("8.8.8.8:80").is_err() {
            return "127.0.0.1".to_string();
        }
        match socket.local_addr() {
            Ok(addr) => addr.ip().to_string(),
            Err(_) => "127.0.0.1".to_string(),
        }
    }

    #[command]
    pub fn get_mobile_prefix() -> String {
        let ip = get_mobile_ip();
        let parts: Vec<&str> = ip.split('.').collect();
        if parts.len() == 4 {
            format!("{}.{}", parts[0], parts[1])
        } else {
            "192.168".to_string()
        }
    }

    #[command]
    pub async fn start_listener_command(app_handle: tauri::AppHandle, seed: u64) -> Result<(), String> {
        start_signal_listener(app_handle, seed);
        Ok(())
    }

    #[command]
    pub async fn send_signal_offer(target_ip: String, seed: u64, sdp: String) -> Result<String, String> {
        let mut stream = TcpStream::connect(format!("{}:9999", target_ip)).await.map_err(|e| e.to_string())?;
        let msg = SignalMessage { seed, sdp };
        let json = serde_json::to_vec(&msg).map_err(|e| e.to_string())?;
        stream.write_all(&json).await.map_err(|e| e.to_string())?;
        
        let mut buf = vec![0u8; 1024 * 16];
        let n = stream.read(&mut buf).await.map_err(|e| e.to_string())?;
        let resp: SignalMessage = serde_json::from_slice(&buf[..n]).map_err(|e| e.to_string())?;
        Ok(resp.sdp)
    }

    #[command]
    pub async fn submit_signal_answer(target_ip: String, sdp: String) -> Result<(), String> {
        let map = PENDING_ANSWERS.lock().await;
        if let Some(tx) = map.get(&target_ip) {
            let _ = tx.send(sdp).await;
            Ok(())
        } else {
            Err(format!("No pending session for IP: {}", target_ip))
        }
    }
}

pub fn start_signal_listener(app_handle: tauri::AppHandle, seed: u64) {
    tokio::spawn(async move {
        let listener = match TcpListener::bind("0.0.0.0:9999").await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[SIGNAL] Failed to bind 9999: {}", e);
                return;
            }
        };
        loop {
            if let Ok((mut socket, addr)) = listener.accept().await {
                let app_handle_clone = app_handle.clone();
                let ip_str = addr.ip().to_string();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 1024 * 16];
                    if let Ok(n) = socket.read(&mut buf).await {
                        if let Ok(msg) = serde_json::from_slice::<SignalMessage>(&buf[..n]) {
                            if msg.seed == seed {
                                let (tx, mut rx) = mpsc::channel::<String>(1);
                                {
                                    let mut map = PENDING_ANSWERS.lock().await;
                                    map.insert(ip_str.clone(), tx);
                                }
                                use tauri::Emitter;
                                let _ = app_handle_clone.emit("webrtc-offer", (msg.sdp, ip_str.clone()));
                                tokio::select! {
                                    Some(answer_sdp) = rx.recv() => {
                                        let resp = SignalMessage { seed, sdp: answer_sdp };
                                        if let Ok(json) = serde_json::to_vec(&resp) {
                                            let _ = socket.write_all(&json).await;
                                        }
                                    }
                                    _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {}
                                }
                                let mut map = PENDING_ANSWERS.lock().await;
                                map.remove(&ip_str);
                            }
                        }
                    }
                });
            }
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::get_deterministic_cert,
            commands::get_mobile_ip,
            commands::get_mobile_prefix,
            commands::start_listener_command,
            commands::send_signal_offer,
            commands::submit_signal_answer
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri mobile application");
}
