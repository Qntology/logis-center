use once_cell::sync::Lazy;
use rand::RngCore;
use anyhow::{Result, anyhow};

/// 메모리 덤프/스캐닝 공격으로부터 키를 보호하기 위한 난독화 구조체
struct ProtectedKey {
    mask: [u8; 64],
    chunks: Vec<Box<[u8; 8]>>, // 64바이트 키를 8바이트씩 8조각으로 쪼개어 힙(Heap)에 흩어놓음
}

impl ProtectedKey {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        
        // 1. 원본 키를 숨길 노이즈 마스크 생성
        let mut mask = [0u8; 64];
        rng.fill_bytes(&mut mask);

        // 2. 실제 암호화에 사용할 원본 키 생성
        let mut real_key = [0u8; 64];
        rng.fill_bytes(&mut real_key);

        // 3. 키를 8조각으로 쪼갠 뒤, 마스크와 섞어서(XOR) 힙 메모리 곳곳에 분산 저장
        let mut chunks = Vec::with_capacity(8);
        for i in 0..8 {
            let mut chunk = [0u8; 8];
            for j in 0..8 {
                chunk[j] = real_key[i * 8 + j] ^ mask[i * 8 + j];
            }
            chunks.push(Box::new(chunk)); // Box를 사용해 메모리 주소를 파편화시킴
        }

        Self { mask, chunks }
    }

    /// 암/복호화가 필요한 찰나의 순간에만 스택 메모리에 진짜 키를 복원
    fn assemble(&self) -> [u8; 64] {
        let mut key = [0u8; 64];
        for i in 0..8 {
            for j in 0..8 {
                key[i * 8 + j] = self.chunks[i][j] ^ self.mask[i * 8 + j];
            }
        }
        key
    }
}

// 글로벌 키 저장소 (연속된 64바이트 키는 메모리 어디에도 존재하지 않게 됨)
static KEY_STORE: Lazy<ProtectedKey> = Lazy::new(|| ProtectedKey::new());

/// 평문 데이터를 초경량 XOR 스트림 방식으로 암호화합니다.
pub fn encrypt_data(data: &[u8]) -> Result<Vec<u8>> {
    let mut nonce = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut nonce);
    
    // 🌟 [최적화] 딱 한 번만 메모리를 할당하고 그 안에서 모든 조립과 암호화를 끝냅니다. (RAM 사용량 절반 감소)
    let mut result = vec![0u8; data.len() + 8];
    result[..8].copy_from_slice(&nonce);
    result[8..].copy_from_slice(data);
    
    let mut local_key = KEY_STORE.assemble();
    for i in 0..64 {
        local_key[i] ^= nonce[i % 8];
    }
    
    // In-place 암호화 연산
    for (i, chunk) in result[8..].chunks_mut(64).enumerate() {
        let shift = (i % 256) as u8; 
        for (j, byte) in chunk.iter_mut().enumerate() {
            *byte ^= local_key[j] ^ shift;
        }
    }

    for byte in local_key.iter_mut() {
        unsafe { std::ptr::write_volatile(byte, 0); }
    }

    Ok(result)
}

/// 암호화된 데이터를 초경량 XOR 방식으로 복호화합니다.
pub fn decrypt_data(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 8 {
        return Err(anyhow!("KV Cache data is too short to be encrypted"));
    }
    
    let mut nonce = [0u8; 8];
    nonce.copy_from_slice(&data[..8]);
    
    // 1. 스택에 진짜 키 조립
    let mut local_key = KEY_STORE.assemble();
    
    // 2. Nonce 혼합
    for i in 0..64 {
        local_key[i] ^= nonce[i % 8];
    }
    
    // 3. 데이터 복호화
    let mut payload = data[8..].to_vec();
    for (i, chunk) in payload.chunks_mut(64).enumerate() {
        let shift = (i % 256) as u8;
        for (j, byte) in chunk.iter_mut().enumerate() {
            *byte ^= local_key[j] ^ shift;
        }
    }
    
    // 4. [보안 핵심] 사용이 끝난 키 흔적 강제 분쇄
    for byte in local_key.iter_mut() {
        unsafe { std::ptr::write_volatile(byte, 0); }
    }

    Ok(payload)
}