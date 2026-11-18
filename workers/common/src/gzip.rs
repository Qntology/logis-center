//! pako 의 `gzip` · `ungzip` 와 `TextEncoder` · `TextDecoder`.
//!
//! 압축된 바이트는 pako 와 같지 않다(zlib 이식 vs miniz_oxide). 풀면 같은 글이고, 저장 · 비교하는 쪽(D1 BLOB, isDiff)은
//! 항상 풀어서 쓰므로 동작에는 차이가 없다.

use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use crate::jsv::{parse_str, stringify_checked, Throw, R, V};

/// `gzip(bytes)`
pub fn gzip(data: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(data).expect("in-memory gzip");
    enc.finish().expect("in-memory gzip")
}

/// `ungzip(x)` — x 는 ArrayBuffer · Uint8Array · 숫자 배열(D1 BLOB). 풀 수 없으면 pako 처럼 던진다.
///
/// 원본 그대로: D1 은 BLOB 을 **숫자 배열**로 돌려주는데, pako 는 배열을 Uint8Array 로 바꾸지 않고 그대로 읽다가 저장 블록(압축하지 않은
/// 블록, 짧거나 잘 줄지 않는 글에 생김)이나 gzip FEXTRA 를 복사할 때 `input.subarray` 를 불러 TypeError 를 던진다. 배열 입력이면 같은
/// 자리에서 던진다 ([`pako_array_throws`]).
pub fn ungzip(v: &V) -> R<Vec<u8>> {
    let bytes = v.to_bytes().ok_or_else(|| Throw::error("invalid input"))?;
    if matches!(v, V::Arr(_)) && pako_array_throws(&bytes) {
        return Err(Throw::type_error("input.subarray is not a function"));
    }
    let mut out = Vec::new();
    GzDecoder::new(&bytes[..])
        .read_to_end(&mut out)
        .map_err(|e| Throw::error(format!("{e}")))?;
    Ok(out)
}

/// pako 가 숫자 배열 입력에서 `input.subarray` 를 부르는가: gzip FEXTRA(길이 > 0) 또는 길이 > 0 인 저장 블록이 있으면 참.
/// 블록 경계를 알려면 허프만 블록도 풀어야 해서 puff(zlib contrib) 꼴의 작은 해독기로 걷는다. 스트림이 깨졌으면 거짓(풀기에서 실패한다)
pub fn pako_array_throws(gz: &[u8]) -> bool {
    walk_gzip(gz).unwrap_or(false)
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
    nbits: u32,
}

impl<'a> Bits<'a> {
    fn need(&mut self, n: u32) -> Option<u32> {
        while self.nbits < n {
            let b = *self.data.get(self.pos)? as u32;
            self.pos += 1;
            self.bit |= b << self.nbits;
            self.nbits += 8;
        }
        let v = self.bit & ((1u32 << n) - 1);
        self.bit >>= n;
        self.nbits -= n;
        Some(v)
    }
    fn align(&mut self) {
        self.bit = 0;
        self.nbits = 0;
    }
}

/// (길이별 부호 수, 기호) — puff 의 canonical 허프만
struct Huff {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Huff {
    fn new(lengths: &[u8]) -> Huff {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        // 길이별 첫 자리 (길이 0 은 부호가 아니다)
        let mut offs = [0u16; 16];
        for len in 1..15 {
            offs[len + 1] = offs[len] + count[len];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (s, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbol[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        Huff { count, symbol }
    }

    fn decode(&self, b: &mut Bits) -> Option<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= b.need(1)? as i32;
            let count = self.count[len] as i32;
            if code - count < first {
                return self.symbol.get((index + (code - first)) as usize).copied();
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT: [u32; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DEXT: [u32; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// 길이 · 거리 기호를 끝(256)까지 읽는다 (출력은 만들지 않는다)
fn codes(b: &mut Bits, lencode: &Huff, distcode: &Huff) -> Option<()> {
    loop {
        let sym = lencode.decode(b)?;
        if sym < 256 {
            continue;
        }
        if sym == 256 {
            return Some(());
        }
        let i = (sym - 257) as usize;
        if i >= 29 {
            return None;
        }
        let _len = LBASE[i] as u32 + b.need(LEXT[i])?;
        let d = distcode.decode(b)? as usize;
        if d >= 30 {
            return None;
        }
        b.need(DEXT[d])?;
    }
}

fn walk_gzip(gz: &[u8]) -> Option<bool> {
    if gz.len() < 18 || gz[0] != 0x1f || gz[1] != 0x8b || gz[2] != 8 {
        return None;
    }
    let flg = gz[3];
    let mut pos = 10;
    if flg & 4 != 0 {
        let xlen = u16::from_le_bytes([*gz.get(pos)?, *gz.get(pos + 1)?]) as usize;
        if xlen > 0 {
            return Some(true);
        }
        pos += 2 + xlen;
    }
    if flg & 8 != 0 {
        pos += gz.get(pos..)?.iter().position(|&c| c == 0)? + 1;
    }
    if flg & 16 != 0 {
        pos += gz.get(pos..)?.iter().position(|&c| c == 0)? + 1;
    }
    if flg & 2 != 0 {
        pos += 2;
    }
    let mut b = Bits { data: gz.get(pos..)?, pos: 0, bit: 0, nbits: 0 };
    loop {
        let last = b.need(1)?;
        match b.need(2)? {
            0 => {
                b.align();
                let d = b.data;
                let len = u16::from_le_bytes([*d.get(b.pos)?, *d.get(b.pos + 1)?]) as usize;
                if len > 0 {
                    return Some(true);
                }
                b.pos += 4 + len;
            }
            1 => {
                let mut lengths = [0u8; 288];
                lengths[..144].fill(8);
                lengths[144..256].fill(9);
                lengths[256..280].fill(7);
                lengths[280..].fill(8);
                let lencode = Huff::new(&lengths);
                let distcode = Huff::new(&[5u8; 30]);
                codes(&mut b, &lencode, &distcode)?;
            }
            2 => {
                let nlen = b.need(5)? as usize + 257;
                let ndist = b.need(5)? as usize + 1;
                let ncode = b.need(4)? as usize + 4;
                const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                let mut lengths = [0u8; 320];
                for &o in ORDER.iter().take(ncode) {
                    lengths[o] = b.need(3)? as u8;
                }
                let lencode = Huff::new(&lengths[..19]);
                let mut index = 0;
                let mut all = [0u8; 320];
                while index < nlen + ndist {
                    let sym = lencode.decode(&mut b)?;
                    if sym < 16 {
                        all[index] = sym as u8;
                        index += 1;
                    } else {
                        let (val, rep) = match sym {
                            16 => (*all.get(index.checked_sub(1)?)?, 3 + b.need(2)? as usize),
                            17 => (0, 3 + b.need(3)? as usize),
                            _ => (0, 11 + b.need(7)? as usize),
                        };
                        if index + rep > nlen + ndist {
                            return None;
                        }
                        all[index..index + rep].fill(val);
                        index += rep;
                    }
                }
                let lencode = Huff::new(&all[..nlen]);
                let distcode = Huff::new(&all[nlen..nlen + ndist]);
                codes(&mut b, &lencode, &distcode)?;
            }
            _ => return None,
        }
        if last == 1 {
            return Some(false);
        }
    }
}

/// `new TextDecoder('utf-8').decode(bytes)` (잘못된 바이트는 U+FFFD, 앞의 BOM 은 뺀다)
pub fn decode_utf8(bytes: &[u8]) -> String {
    let b = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    String::from_utf8_lossy(b).into_owned()
}

/// `JSON.parse(new TextDecoder('utf-8').decode(ungzip(v)))` — 원본이 수십 번 쓰는 꼴
pub fn ungzip_json(v: &V) -> R<V> {
    parse_str(&decode_utf8(&ungzip(v)?))
}

/// `gzip(new TextEncoder('utf-8').encode(JSON.stringify(x)), …).buffer` — D1 에 넣는 ArrayBuffer.
/// `JSON.stringify(undefined)` 는 undefined 라 TextEncoder 가 "undefined" 를 인코딩한다(원본 그대로)
pub fn gzip_json(v: &V) -> R<V> {
    let text = stringify_checked(v)?.unwrap_or_else(|| "undefined".into());
    Ok(V::bytes(gzip(text.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obj;

    #[test]
    fn round_trip_through_a_d1_number_array() {
        let x = obj! { "text" => "안녕", "n" => 1.5 };
        let V::Bytes(b) = gzip_json(&x).unwrap() else { panic!() };
        // D1 은 BLOB 을 숫자 배열로 돌려준다
        let as_array = V::from(b.borrow().iter().map(|x| V::Num(*x as f64)).collect::<Vec<_>>());
        let back = ungzip_json(&as_array).unwrap();
        assert_eq!(crate::jsv::stringify(&back).unwrap(), r#"{"text":"안녕","n":1.5}"#);
        assert!(ungzip(&V::from("not gzip")).is_err());
        assert_eq!(decode_utf8(&[0xEF, 0xBB, 0xBF, b'a']), "a");
    }
}
