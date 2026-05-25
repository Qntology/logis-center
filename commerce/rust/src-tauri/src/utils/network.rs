use rcgen::{CertificateParams, KeyPair, PKCS_ECDSA_P256_SHA256, DistinguishedName};
use rand_chacha::ChaCha20Rng;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

// [NEW] 대기 중인 응답 소켓들을 관리하기 위한 글로벌 맵
static PENDING_ANSWERS: Lazy<Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(HashMap::new()))
});

static ACTIVE_SEED: AtomicU64 = AtomicU64::new(0);
static LISTENER_STARTED: AtomicBool = AtomicBool::new(false);

#[derive(Serialize, Deserialize, Debug)]
pub struct SignalMessage {
    pub seed: u64,
    pub sdp: String,
}

pub fn get_deterministic_cert(seed_num: u64) -> (String, String) {
    use rand::SeedableRng;
    let _rng = ChaCha20Rng::seed_from_u64(seed_num);
    let key_pair = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
    let mut params = CertificateParams::new(vec!["LocalNode".to_string()]).unwrap();
    params.distinguished_name = DistinguishedName::new();
    params.distinguished_name.push(rcgen::DnType::CommonName, "LocalNode");
    let cert = params.self_signed(&key_pair).unwrap();
    (cert.pem(), key_pair.serialize_pem())
}

pub fn get_my_full_ip() -> String {
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

pub fn get_local_network_prefix() -> String {
    let ip = get_my_full_ip();
    let parts: Vec<&str> = ip.split('.').collect();
    if parts.len() == 4 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        "192.168".to_string()
    }
}

pub fn start_signal_listener(seed: u64) {
    ACTIVE_SEED.store(seed, Ordering::SeqCst);
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
                let ip_str = addr.ip().to_string();
                
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 1024 * 16];
                    if let Ok(n) = socket.read(&mut buf).await {
                        if let Ok(msg) = serde_json::from_slice::<SignalMessage>(&buf[..n]) {
                            let current_seed = ACTIVE_SEED.load(Ordering::SeqCst);
                            if msg.seed == current_seed {
                                println!("[SIGNAL] Seed match from {}. Offer received.", ip_str);
                                
                                let (tx, mut rx) = mpsc::channel::<String>(1);
                                {
                                    let mut map = PENDING_ANSWERS.lock().await;
                                    map.insert(ip_str.clone(), tx);
                                }

                                println!("[SIGNAL] WebRTC Offer SDP Received from {}", ip_str);

                                tokio::select! {
                                    Some(answer_sdp) = rx.recv() => {
                                        let resp = SignalMessage { seed, sdp: answer_sdp };
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

pub async fn send_signal_offer(target_ip: String, seed: u64, sdp: String) -> Result<String, String> {
    let mut stream = TcpStream::connect(format!("{}:9999", target_ip)).await.map_err(|e| e.to_string())?;
    let msg = SignalMessage { seed, sdp };
    let json = serde_json::to_vec(&msg).map_err(|e| e.to_string())?;
    stream.write_all(&json).await.map_err(|e| e.to_string())?;
    
    // 상대방의 Answer 대기
    let mut buf = vec![0u8; 1024 * 16];
    let n = stream.read(&mut buf).await.map_err(|e| e.to_string())?;
    let resp: SignalMessage = serde_json::from_slice(&buf[..n]).map_err(|e| e.to_string())?;
    
    Ok(resp.sdp)
}

pub async fn submit_signal_answer(target_ip: String, sdp: String) -> Result<(), String> {
    let map = PENDING_ANSWERS.lock().await;
    if let Some(tx) = map.get(&target_ip) {
        let _ = tx.send(sdp).await;
        Ok(())
    } else {
        Err(format!("No pending session for IP: {}", target_ip))
    }
}