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
// 🌟 [PARITY / network.rs] 실시간 시드 변경 및 포트 중복(10048) 차단을 위한 Atomic 패키지
//    ── 왜 필요한가 ──
//     initManualConnectUI() 는 앱 부팅뿐 아니라 시드 충돌 자동 양보 시에도
//     start_listener_command 를 재호출합니다. 데스크톱(network.rs)은 이미
//     LISTENER_STARTED 로 재바인딩을 막고 ACTIVE_SEED 만 갱신하는데,
//     모바일에는 두 장치가 모두 없어 두 번째 호출부터 9999 포트 바인딩이 실패하고
//     (os error 10048 / EADDRINUSE) 시드도 최초 값에 영구 고정되었습니다.
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
// [NEW] 대기 중인 응답 소켓들을 관리하기 위한 글로벌 맵
static PENDING_ANSWERS: Lazy<Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(HashMap::new()))
});
// 🌟 [PARITY / network.rs] 백엔드가 실시간으로 바라볼 시드 저장소 및 리스너 생존 상태
static ACTIVE_SEED: AtomicU64 = AtomicU64::new(0);
static LISTENER_STARTED: AtomicBool = AtomicBool::new(false);
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

    // 🌟 [CRITICAL FIX] createPeerConnection() 은 데스크톱과 동일한 이름
    //    invoke("get_my_full_ip") 를 호출하는데, 모바일에는 그 커맨드가 없어
    //    Answer QR 생성 직전에 예외가 던져지고 연결이 100% 실패했습니다.
    //    (로그: "Err: Command get_my_full_ip not found")
    //    호출부를 고치는 대신 양쪽 이름을 모두 등록해 두면
    //    이후 데스크톱 코드를 그대로 이식해도 깨지지 않습니다.
    #[command]
    pub fn get_my_full_ip() -> String {
        get_mobile_ip()
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

    // 🌟 [PARITY / network.rs] 데스크톱의 get_local_network_prefix 와 동일한 별칭입니다.
    #[command]
    pub fn get_local_network_prefix() -> String {
        get_mobile_prefix()
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

// 🌟 [PARITY / network.rs] TCP 시그널링 리스너 (Answer 회신 대기 기능 포함)
pub fn start_signal_listener(app_handle: tauri::AppHandle, seed: u64) {
    // 🌟 [CRITICAL FIX] 명령이 들어올 때마다 시드(Seed) 번호만 최신화합니다.
    ACTIVE_SEED.store(seed, Ordering::SeqCst);

    // 🌟 [CRITICAL FIX] 이미 9999 포트가 열려있다면, 포트 충돌(os error 10048)을 방지하기 위해 여기서 튕겨냅니다.
    if LISTENER_STARTED.swap(true, Ordering::SeqCst) {
        println!("[SIGNAL] Listener already running. Seed updated to: {}", seed);
        return;
    }

    tokio::spawn(async move {
        let listener = match TcpListener::bind("0.0.0.0:9999").await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[SIGNAL] Failed to bind 9999: {}", e);
                LISTENER_STARTED.store(false, Ordering::SeqCst);
                return;
            }
        };
        println!("[SIGNAL] Listening on 9999 for seed: {}", seed);
        loop {
            if let Ok((mut socket, addr)) = listener.accept().await {
                let app_handle_clone = app_handle.clone();
                let ip_str = addr.ip().to_string();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 1024 * 16];
                    if let Ok(n) = socket.read(&mut buf).await {
                        if let Ok(msg) = serde_json::from_slice::<SignalMessage>(&buf[..n]) {
                            // 🌟 [CRITICAL FIX] 고정된 매개변수가 아닌 실시간으로 업데이트되는 ACTIVE_SEED를 검사합니다!
                            let current_seed = ACTIVE_SEED.load(Ordering::SeqCst);
                            if msg.seed == current_seed {
                                println!("[SIGNAL] Seed match from {}. Offer received.", ip_str);

                                let (tx, mut rx) = mpsc::channel::<String>(1);
                                {
                                    let mut map = PENDING_ANSWERS.lock().await;
                                    map.insert(ip_str.clone(), tx);
                                }
                                use tauri::Emitter;
                                let _ = app_handle_clone.emit("webrtc-offer", (msg.sdp, ip_str.clone()));
                                tokio::select! {
                                    Some(answer_sdp) = rx.recv() => {
                                        // 🌟 회신 시에도 실시간 시드를 실어 보내야 상대방의 검증을 통과합니다.
                                        let resp = SignalMessage { seed: current_seed, sdp: answer_sdp };
                                        if let Ok(json) = serde_json::to_vec(&resp) {
                                            let _ = socket.write_all(&json).await;
                                            println!("[SIGNAL] Answer sent back to {}", ip_str);
                                        }
                                    }
                                    _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                                        println!("[SIGNAL] Timeout waiting for answer from {}", ip_str);
                                    }
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
            // 🌟 [CRITICAL FIX] createPeerConnection() 의 실제 호출 이름입니다.
            //    누락 시 Answer QR 이 생성되지 않아 페어링이 영구 실패합니다.
            commands::get_my_full_ip,
            commands::get_mobile_prefix,
            // 🌟 [PARITY] 데스크톱과 동일한 이름의 별칭
            commands::get_local_network_prefix,
            commands::start_listener_command,
            commands::send_signal_offer,
            commands::submit_signal_answer
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri mobile application");
}
