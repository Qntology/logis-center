//! JS 값 모형 — 원본 TypeScript 워커의 동작을 그대로 옮기기 위한 최소한의 JS 의미론.
//!
//! - 객체 · 배열 · 바이트는 참조를 공유한다(`Rc<RefCell<…>>`): `var page = pages[i]; page.x = 1` 이 `pages[i]` 를 바꾸는 것까지 같다.
//!   복제가 필요하면 [`V::deep_clone`](safeClone · 구조 복사).
//! - `undefined`([`V::Undef`]) 와 `null`([`V::Null`]) 을 나눈다. `JSON.stringify` 는 객체의 undefined 속성을 빼고 배열에선 null 로 쓴다.
//! - 속성 읽기 `a.b` 는 [`V::g`]: `a` 가 undefined · null 이면 JS 처럼 TypeError 를 던진다([`Throw`]). 원본이 try/catch 로 그 예외를
//!   삼키는 곳은 `match`/`if let Err` 로, 함수 밖으로 새는 곳은 `?` 로 옮긴다. 던지지 않는 읽기는 [`V::get`].
//! - 키 순서: JS 와 같이 배열 첨자 꼴 키(0, 1, 2…)가 오름차순으로 먼저, 나머지는 넣은 순서([`V::keys`]).
//! - 수: 모두 f64. 문자열로 바꿀 때는 JS `Number#toString` 과 같은 꼴(ryu-js).
//! - 문자열 길이 · 위치는 JS 처럼 UTF-16 단위([`u16_len`] · [`index_of`] · [`substring`] · [`slice`]).
//! - [`V::Ext`] 는 JS 쪽 객체(D1 준비된 문장 등)를 그대로 들고 다니는 칸이다. `JSON.stringify` 에선 `{}`.

use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use indexmap::IndexMap;

pub type Map = IndexMap<String, V>;

/// JS 예외 (`TypeError: …` · `SyntaxError: …` · `Error: …`)
#[derive(Debug, Clone, PartialEq)]
pub struct Throw {
    pub name: &'static str,
    pub message: String,
}

impl Throw {
    pub fn type_error(message: impl Into<String>) -> Throw {
        Throw { name: "TypeError", message: message.into() }
    }
    pub fn syntax_error(message: impl Into<String>) -> Throw {
        Throw { name: "SyntaxError", message: message.into() }
    }
    pub fn range_error(message: impl Into<String>) -> Throw {
        Throw { name: "RangeError", message: message.into() }
    }
    pub fn error(message: impl Into<String>) -> Throw {
        Throw { name: "Error", message: message.into() }
    }
}

impl fmt::Display for Throw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.message)
    }
}

pub type R<T> = Result<T, Throw>;

#[derive(Clone, Default)]
pub enum V {
    #[default]
    Undef,
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Rc<RefCell<Vec<V>>>),
    Obj(Rc<RefCell<Map>>),
    /// ArrayBuffer · Uint8Array
    Bytes(Rc<RefCell<Vec<u8>>>),
    /// JS 쪽 객체 (D1PreparedStatement 등). 값 비교는 같은 참조일 때만 같다
    Ext(Rc<dyn Any>),
}

impl fmt::Debug for V {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            V::Undef => write!(f, "undefined"),
            V::Ext(_) => write!(f, "[object Ext]"),
            V::Bytes(b) => write!(f, "ArrayBuffer({})", b.borrow().len()),
            other => write!(f, "{}", stringify(other).unwrap_or_else(|| "undefined".into())),
        }
    }
}

impl PartialEq for V {
    /// `===` (객체는 같은 참조)
    fn eq(&self, other: &V) -> bool {
        strict_eq(self, other)
    }
}

impl From<f64> for V {
    fn from(n: f64) -> V {
        V::Num(n)
    }
}
impl From<i64> for V {
    fn from(n: i64) -> V {
        V::Num(n as f64)
    }
}
impl From<i32> for V {
    fn from(n: i32) -> V {
        V::Num(n as f64)
    }
}
impl From<u32> for V {
    fn from(n: u32) -> V {
        V::Num(n as f64)
    }
}
impl From<usize> for V {
    fn from(n: usize) -> V {
        V::Num(n as f64)
    }
}
impl From<bool> for V {
    fn from(b: bool) -> V {
        V::Bool(b)
    }
}
impl From<&str> for V {
    fn from(s: &str) -> V {
        V::Str(s.to_string())
    }
}
impl From<String> for V {
    fn from(s: String) -> V {
        V::Str(s)
    }
}
impl From<&String> for V {
    fn from(s: &String) -> V {
        V::Str(s.clone())
    }
}
impl From<Vec<V>> for V {
    fn from(items: Vec<V>) -> V {
        V::Arr(Rc::new(RefCell::new(items)))
    }
}
impl<T: Into<V>> From<Option<T>> for V {
    fn from(o: Option<T>) -> V {
        o.map_or(V::Undef, Into::into)
    }
}

/// `{ "a": 1, "b": x }` — 넣은 순서 그대로
#[macro_export]
macro_rules! obj {
    () => { $crate::jsv::V::new_obj() };
    ($($k:expr => $v:expr),+ $(,)?) => {{
        let o = $crate::jsv::V::new_obj();
        $( o.put($k, $crate::jsv::V::from($v)); )+
        o
    }};
}

/// `[a, b, c]`
#[macro_export]
macro_rules! arr {
    () => { $crate::jsv::V::new_arr() };
    ($($v:expr),+ $(,)?) => { $crate::jsv::V::from(vec![$($crate::jsv::V::from($v)),+]) };
}

/// 배열 첨자 꼴 키 ("0", "12", 앞자리 0 없음, 2^32-1 미만)
pub fn array_index(key: &str) -> Option<u32> {
    if key.is_empty() || key.len() > 10 || !key.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if key.len() > 1 && key.starts_with('0') {
        return None;
    }
    let n: u64 = key.parse().ok()?;
    (n < u32::MAX as u64).then_some(n as u32)
}

impl V {
    pub fn new_obj() -> V {
        V::Obj(Rc::new(RefCell::new(Map::new())))
    }
    pub fn new_arr() -> V {
        V::Arr(Rc::new(RefCell::new(Vec::new())))
    }
    pub fn bytes(b: Vec<u8>) -> V {
        V::Bytes(Rc::new(RefCell::new(b)))
    }
    pub fn ext<T: Any>(t: T) -> V {
        V::Ext(Rc::new(t))
    }
    pub fn s(text: impl Into<String>) -> V {
        V::Str(text.into())
    }
    pub fn n(x: impl Into<f64>) -> V {
        V::Num(x.into())
    }

    pub fn is_undef(&self) -> bool {
        matches!(self, V::Undef)
    }
    pub fn is_null(&self) -> bool {
        matches!(self, V::Null)
    }
    /// `x == null`
    pub fn is_nullish(&self) -> bool {
        matches!(self, V::Undef | V::Null)
    }
    pub fn is_arr(&self) -> bool {
        matches!(self, V::Arr(_))
    }
    pub fn is_obj(&self) -> bool {
        matches!(self, V::Obj(_))
    }
    pub fn is_str(&self) -> bool {
        matches!(self, V::Str(_))
    }
    pub fn is_num(&self) -> bool {
        matches!(self, V::Num(_))
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            V::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_num(&self) -> Option<f64> {
        match self {
            V::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn ext_ref<T: Any>(&self) -> Option<&T> {
        match self {
            V::Ext(e) => e.downcast_ref::<T>(),
            _ => None,
        }
    }

    /// `typeof x`
    pub fn type_of(&self) -> &'static str {
        match self {
            V::Undef => "undefined",
            V::Bool(_) => "boolean",
            V::Num(_) => "number",
            V::Str(_) => "string",
            V::Null | V::Arr(_) | V::Obj(_) | V::Bytes(_) | V::Ext(_) => "object",
        }
    }

    /// `if (x)`
    pub fn truthy(&self) -> bool {
        match self {
            V::Undef | V::Null => false,
            V::Bool(b) => *b,
            V::Num(n) => *n != 0.0 && !n.is_nan(),
            V::Str(s) => !s.is_empty(),
            _ => true,
        }
    }

    /// `String(x)` · 템플릿 `${x}`
    pub fn to_str(&self) -> String {
        match self {
            V::Undef => "undefined".into(),
            V::Null => "null".into(),
            V::Bool(b) => b.to_string(),
            V::Num(n) => num_to_string(*n),
            V::Str(s) => s.clone(),
            V::Arr(a) => a
                .borrow()
                .iter()
                .map(|x| if x.is_nullish() { String::new() } else { x.to_str() })
                .collect::<Vec<_>>()
                .join(","),
            V::Obj(_) => "[object Object]".into(),
            V::Bytes(_) => "[object ArrayBuffer]".into(),
            V::Ext(_) => "[object Object]".into(),
        }
    }

    /// `Number(x)` · 단항 `+x`
    pub fn to_num(&self) -> f64 {
        match self {
            V::Undef => f64::NAN,
            V::Null => 0.0,
            V::Bool(b) => *b as u8 as f64,
            V::Num(n) => *n,
            V::Str(s) => string_to_number(s),
            V::Arr(_) => string_to_number(&self.to_str()),
            _ => f64::NAN,
        }
    }

    /// 프로퍼티 키로서의 문자열 (`obj[x]`)
    pub fn to_key(&self) -> String {
        self.to_str()
    }

    /// 던지지 않는 읽기: 없거나 객체가 아니면 undefined. `length`(배열 · 문자열) · `byteLength` 도 읽는다
    pub fn get(&self, key: &str) -> V {
        match self {
            V::Obj(o) => o.borrow().get(key).cloned().unwrap_or(V::Undef),
            V::Arr(a) => {
                if key == "length" {
                    return V::Num(a.borrow().len() as f64);
                }
                match array_index(key) {
                    Some(i) => a.borrow().get(i as usize).cloned().unwrap_or(V::Undef),
                    None => V::Undef,
                }
            }
            V::Str(s) => {
                if key == "length" {
                    return V::Num(u16_len(s) as f64);
                }
                match array_index(key) {
                    Some(i) => char_at(s, i as usize).map_or(V::Undef, V::Str),
                    None => V::Undef,
                }
            }
            V::Bytes(b) => match key {
                "byteLength" | "length" => V::Num(b.borrow().len() as f64),
                _ => array_index(key)
                    .and_then(|i| b.borrow().get(i as usize).map(|x| V::Num(*x as f64)))
                    .unwrap_or(V::Undef),
            },
            _ => V::Undef,
        }
    }

    /// `a.b` — a 가 undefined · null 이면 TypeError
    pub fn g(&self, key: &str) -> R<V> {
        match self {
            V::Undef | V::Null => Err(Throw::type_error(format!(
                "Cannot read properties of {} (reading '{key}')",
                self.to_str()
            ))),
            _ => Ok(self.get(key)),
        }
    }

    /// `a[k]` (계산된 키)
    pub fn gk(&self, key: &V) -> R<V> {
        self.g(&key.to_key())
    }

    /// `a[i]`
    pub fn gi(&self, i: usize) -> R<V> {
        self.g(&i.to_string())
    }

    /// 던지지 않는 첨자 읽기
    pub fn at(&self, i: usize) -> V {
        self.get(&i.to_string())
    }

    /// `a.b = v` — 원본은 ES 모듈이라 엄격 모드다: a 가 undefined · null 이면 TypeError, 원시값(문자열 · 수 · 참거짓)에 쓰면
    /// 그것도 TypeError (`Cannot create property 'b' on string '…'`)
    pub fn s_(&self, key: &str, val: V) -> R<()> {
        match self {
            V::Undef | V::Null => Err(Throw::type_error(format!(
                "Cannot set properties of {} (setting '{key}')",
                self.to_str()
            ))),
            V::Str(_) | V::Num(_) | V::Bool(_) => Err(Throw::type_error(format!(
                "Cannot create property '{key}' on {} '{}'",
                self.type_of(),
                self.to_str()
            ))),
            _ => {
                self.put(key, val);
                Ok(())
            }
        }
    }

    /// `a[k] = v`
    pub fn sk(&self, key: &V, val: V) -> R<()> {
        self.s_(&key.to_key(), val)
    }

    /// 던지지 않는 쓰기 (객체 · 배열에만 효과)
    pub fn put(&self, key: &str, val: V) {
        match self {
            V::Obj(o) => {
                o.borrow_mut().insert(key.to_string(), val);
            }
            V::Arr(a) => {
                if key == "length" {
                    let n = val.to_num();
                    if n >= 0.0 && n.fract() == 0.0 {
                        a.borrow_mut().resize(n as usize, V::Undef);
                    }
                    return;
                }
                if let Some(i) = array_index(key) {
                    let mut a = a.borrow_mut();
                    let i = i as usize;
                    if i >= a.len() {
                        a.resize(i + 1, V::Undef);
                    }
                    a[i] = val;
                }
                // 배열의 이름 있는 속성은 이 모형에 없다 (원본이 쓰지 않음)
            }
            V::Bytes(b) => {
                if let Some(i) = array_index(key) {
                    let mut b = b.borrow_mut();
                    if (i as usize) < b.len() {
                        b[i as usize] = to_uint8(val.to_num());
                    }
                }
            }
            _ => {}
        }
    }

    /// `delete a.b` — a 가 undefined · null 이면 TypeError
    pub fn del(&self, key: &str) -> R<()> {
        match self {
            V::Undef | V::Null => Err(Throw::type_error(format!(
                "Cannot convert undefined or null to object (deleting '{key}')"
            ))),
            V::Obj(o) => {
                o.borrow_mut().shift_remove(key);
                Ok(())
            }
            V::Arr(a) => {
                if let Some(i) = array_index(key) {
                    if let Some(slot) = a.borrow_mut().get_mut(i as usize) {
                        *slot = V::Undef;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// `obj.hasOwnProperty(k)` · `k in obj`
    pub fn has(&self, key: &str) -> bool {
        match self {
            V::Obj(o) => o.borrow().contains_key(key),
            V::Arr(a) => key == "length" || array_index(key).is_some_and(|i| (i as usize) < a.borrow().len()),
            V::Str(s) => key == "length" || array_index(key).is_some_and(|i| (i as usize) < u16_len(s)),
            _ => false,
        }
    }

    /// `arr.push(v)` — 배열이 아니면 TypeError (`x.push is not a function`)
    pub fn push(&self, val: V) -> R<()> {
        match self {
            V::Arr(a) => {
                a.borrow_mut().push(val);
                Ok(())
            }
            V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'push')", self.to_str()))),
            _ => Err(Throw::type_error("push is not a function")),
        }
    }

    /// 배열 길이 (배열이 아니면 0)
    pub fn len(&self) -> usize {
        match self {
            V::Arr(a) => a.borrow().len(),
            V::Str(s) => u16_len(s),
            V::Bytes(b) => b.borrow().len(),
            _ => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `x.length` — undefined · null 이면 TypeError, 길이가 없으면 undefined
    pub fn length(&self) -> R<V> {
        self.g("length")
    }

    /// 배열 원소의 사본 목록 (원소 자체는 참조 공유)
    pub fn items(&self) -> Vec<V> {
        match self {
            V::Arr(a) => a.borrow().clone(),
            _ => Vec::new(),
        }
    }

    /// `Object.keys(x)` · `for (k in x)` 순서 — undefined · null 이면 TypeError
    pub fn keys_checked(&self) -> R<Vec<String>> {
        match self {
            V::Undef | V::Null => Err(Throw::type_error("Cannot convert undefined or null to object")),
            _ => Ok(self.keys()),
        }
    }

    /// `Object.keys(x)` 순서 (첨자 꼴 키 오름차순, 그다음 넣은 순서). 객체가 아니면 빈 목록
    pub fn keys(&self) -> Vec<String> {
        match self {
            V::Obj(o) => {
                let o = o.borrow();
                let mut idx: Vec<(u32, &String)> = o.keys().filter_map(|k| array_index(k).map(|i| (i, k))).collect();
                idx.sort_by_key(|(i, _)| *i);
                let mut out: Vec<String> = idx.into_iter().map(|(_, k)| k.clone()).collect();
                out.extend(o.keys().filter(|k| array_index(k).is_none()).cloned());
                out
            }
            V::Arr(a) => (0..a.borrow().len()).map(|i| i.to_string()).collect(),
            V::Str(s) => (0..u16_len(s)).map(|i| i.to_string()).collect(),
            V::Bytes(b) => (0..b.borrow().len()).map(|i| i.to_string()).collect(),
            _ => Vec::new(),
        }
    }

    /// (키, 값) — [`V::keys`] 순서
    pub fn entries(&self) -> Vec<(String, V)> {
        self.keys().into_iter().map(|k| {
            let v = self.get(&k);
            (k, v)
        }).collect()
    }

    /// 깊은 복사 (`safeClone`: 같은 객체를 다시 만나면 null). 바이트는 원본 safeClone 처럼 `{}` 가 아니라
    /// 바이트 그대로 복사한다 — D1 이 BLOB 을 숫자 배열로 돌려주므로 원본에서 이 경우는 생기지 않는다
    pub fn deep_clone(&self) -> V {
        fn go(v: &V, seen: &mut Vec<*const ()>) -> V {
            match v {
                V::Arr(a) => {
                    let p = Rc::as_ptr(a) as *const ();
                    if seen.contains(&p) {
                        return V::Null;
                    }
                    seen.push(p);
                    let items: Vec<V> = a.borrow().iter().map(|x| go(x, seen)).collect();
                    V::from(items)
                }
                V::Obj(o) => {
                    let p = Rc::as_ptr(o) as *const ();
                    if seen.contains(&p) {
                        return V::Null;
                    }
                    seen.push(p);
                    let out = V::new_obj();
                    for k in v.keys() {
                        let x = o.borrow().get(&k).cloned().unwrap_or_default();
                        out.put(&k, go(&x, seen));
                    }
                    out
                }
                V::Bytes(b) => V::bytes(b.borrow().clone()),
                other => other.clone(),
            }
        }
        go(self, &mut Vec::new())
    }

    /// 바이트로 (Uint8Array · ArrayBuffer · 숫자 배열 — pako 가 받는 입력)
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        match self {
            V::Bytes(b) => Some(b.borrow().clone()),
            V::Arr(a) => Some(a.borrow().iter().map(|x| to_uint8(x.to_num())).collect()),
            V::Str(s) => Some(s.chars().map(|c| c as u32 as u8).collect()),
            _ => None,
        }
    }
}

/// `new Uint8Array([x])[0]`
pub fn to_uint8(n: f64) -> u8 {
    if !n.is_finite() {
        return 0;
    }
    (n.trunc().rem_euclid(256.0)) as u8
}

/// `===`
pub fn strict_eq(a: &V, b: &V) -> bool {
    match (a, b) {
        (V::Undef, V::Undef) | (V::Null, V::Null) => true,
        (V::Bool(x), V::Bool(y)) => x == y,
        (V::Num(x), V::Num(y)) => x == y,
        (V::Str(x), V::Str(y)) => x == y,
        (V::Arr(x), V::Arr(y)) => Rc::ptr_eq(x, y),
        (V::Obj(x), V::Obj(y)) => Rc::ptr_eq(x, y),
        (V::Bytes(x), V::Bytes(y)) => Rc::ptr_eq(x, y),
        (V::Ext(x), V::Ext(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

/// `==` (느슨한 비교)
pub fn loose_eq(a: &V, b: &V) -> bool {
    use V::*;
    match (a, b) {
        (Undef | Null, Undef | Null) => true,
        (Undef | Null, _) | (_, Undef | Null) => false,
        (Num(_), Num(_)) | (Str(_), Str(_)) | (Bool(_), Bool(_)) => strict_eq(a, b),
        (Num(x), Str(s)) | (Str(s), Num(x)) => *x == string_to_number(s),
        (Bool(x), _) => loose_eq(&Num(*x as u8 as f64), b),
        (_, Bool(y)) => loose_eq(a, &Num(*y as u8 as f64)),
        // 객체 ↔ 원시값: ToPrimitive (배열은 join, 객체는 "[object Object]")
        (Arr(_) | Obj(_) | Bytes(_) | Ext(_), Num(_) | Str(_)) => loose_eq(&Str(a.to_str()), b),
        (Num(_) | Str(_), Arr(_) | Obj(_) | Bytes(_) | Ext(_)) => loose_eq(a, &Str(b.to_str())),
        _ => strict_eq(a, b),
    }
}

/// `a + b`
pub fn add(a: &V, b: &V) -> V {
    let prim = |v: &V| match v {
        V::Arr(_) | V::Obj(_) | V::Bytes(_) | V::Ext(_) => V::Str(v.to_str()),
        other => other.clone(),
    };
    let (pa, pb) = (prim(a), prim(b));
    if pa.is_str() || pb.is_str() {
        V::Str(pa.to_str() + &pb.to_str())
    } else {
        V::Num(pa.to_num() + pb.to_num())
    }
}

/// `a < b` 등 관계 비교: 먼저 ToPrimitive(배열 · 객체는 문자열), 둘 다 문자열이면 UTF-16 사전순, 아니면 수
pub fn less_than(a: &V, b: &V) -> Option<bool> {
    let prim = |v: &V| match v {
        V::Arr(_) | V::Obj(_) | V::Bytes(_) | V::Ext(_) => V::Str(v.to_str()),
        other => other.clone(),
    };
    let (a, b) = (&prim(a), &prim(b));
    if let (V::Str(x), V::Str(y)) = (a, b) {
        return Some(x.encode_utf16().lt(y.encode_utf16()));
    }
    let (x, y) = (a.to_num(), b.to_num());
    if x.is_nan() || y.is_nan() {
        None
    } else {
        Some(x < y)
    }
}

/// JS `Number#toString()`
pub fn num_to_string(n: f64) -> String {
    if n.is_nan() {
        return "NaN".into();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if n == 0.0 {
        return "0".into();
    }
    let mut buf = ryu_js::Buffer::new();
    buf.format(n).to_string()
}

/// JS `StringToNumber` (`Number("…")`)
pub fn string_to_number(s: &str) -> f64 {
    let t = js_trim(s);
    if t.is_empty() {
        return 0.0;
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    let lower = t.to_ascii_lowercase();
    for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if let Some(digits) = lower.strip_prefix(prefix) {
            if !digits.is_empty() && digits.chars().all(|c| c.is_digit(radix)) {
                return digits.chars().fold(0.0, |acc, c| acc * radix as f64 + c.to_digit(radix).unwrap() as f64);
            }
            return f64::NAN;
        }
    }
    // 10진: 부호, 숫자, 소수점, 지수만
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    let valid = !body.is_empty()
        && body.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
        && body.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '.')
        && !body.contains("inf")
        && !body.contains("nan");
    if !valid {
        return f64::NAN;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

/// `parseInt(s)` (기수 10, 앞 공백 무시, 숫자가 끊기는 곳까지)
pub fn parse_int(s: &str) -> f64 {
    let t = js_trim_start(s);
    let (neg, rest) = match t.chars().next() {
        Some('-') => (true, &t[1..]),
        Some('+') => (false, &t[1..]),
        _ => (false, t),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return f64::NAN;
    }
    let v: f64 = digits.parse().unwrap_or(f64::NAN);
    if neg { -v } else { v }
}

/// `parseFloat(s)`
pub fn parse_float(s: &str) -> f64 {
    let t = js_trim_start(s);
    if t.starts_with("Infinity") || t.starts_with("+Infinity") {
        return f64::INFINITY;
    }
    if t.starts_with("-Infinity") {
        return f64::NEG_INFINITY;
    }
    // 가장 긴 10진 접두
    let b = t.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let start_digits = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i == start_digits || (i == start_digits + 1 && b[start_digits] == b'.') {
        return f64::NAN;
    }
    let mut end = i;
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let ds = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > ds {
            end = j;
        }
    }
    t[..end].parse().unwrap_or(f64::NAN)
}

/// JS 의 공백 (WhiteSpace + LineTerminator)
pub fn is_js_space(c: char) -> bool {
    matches!(
        c,
        '\u{0009}' | '\u{000A}' | '\u{000B}' | '\u{000C}' | '\u{000D}' | '\u{0020}' | '\u{00A0}' | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}' | '\u{FEFF}'
    )
}

/// `s.trim()`
pub fn js_trim(s: &str) -> &str {
    s.trim_matches(is_js_space)
}

pub fn js_trim_start(s: &str) -> &str {
    s.trim_start_matches(is_js_space)
}

pub fn js_trim_end(s: &str) -> &str {
    s.trim_end_matches(is_js_space)
}

/// `s.length` (UTF-16 단위)
pub fn u16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `s.charAt(i)` · `s[i]` (서로게이트 쌍의 반쪽은 대체 문자)
pub fn char_at(s: &str, i: usize) -> Option<String> {
    let u: Vec<u16> = s.encode_utf16().collect();
    u.get(i).map(|c| String::from_utf16_lossy(&[*c]))
}

/// `s.indexOf(needle)` (UTF-16 위치, 없으면 -1)
pub fn index_of(s: &str, needle: &str) -> i64 {
    match s.find(needle) {
        Some(byte) => u16_len(&s[..byte]) as i64,
        None => -1,
    }
}

/// `s.includes(needle)`
pub fn includes(s: &str, needle: &str) -> bool {
    s.contains(needle)
}

fn u16_to_byte(s: &str, unit: usize) -> usize {
    let mut count = 0;
    for (byte, c) in s.char_indices() {
        if count >= unit {
            return byte;
        }
        count += c.len_utf16();
    }
    s.len()
}

/// `s.substring(a, b)` (UTF-16, 범위 밖은 자르고, a > b 면 바꿈)
pub fn substring(s: &str, a: f64, b: Option<f64>) -> String {
    let len = u16_len(s) as f64;
    let clamp = |x: f64| if x.is_nan() { 0.0 } else { x.max(0.0).min(len) };
    let (mut x, mut y) = (clamp(a), clamp(b.unwrap_or(len)));
    if x > y {
        std::mem::swap(&mut x, &mut y);
    }
    let (bx, by) = (u16_to_byte(s, x as usize), u16_to_byte(s, y as usize));
    s[bx..by].to_string()
}

/// `s.slice(a, b)` (UTF-16, 음수는 끝에서)
pub fn slice(s: &str, a: f64, b: Option<f64>) -> String {
    let len = u16_len(s) as f64;
    let norm = |x: f64| {
        if x.is_nan() {
            0.0
        } else if x < 0.0 {
            (len + x.trunc()).max(0.0)
        } else {
            x.trunc().min(len)
        }
    };
    let (x, y) = (norm(a), norm(b.unwrap_or(len)));
    if x >= y {
        return String::new();
    }
    let (bx, by) = (u16_to_byte(s, x as usize), u16_to_byte(s, y as usize));
    s[bx..by].to_string()
}

/// `s.split(sep)` (문자열 구분자)
pub fn split(s: &str, sep: &str) -> Vec<String> {
    if sep.is_empty() {
        return s.encode_utf16().map(|u| String::from_utf16_lossy(&[u])).collect();
    }
    s.split(sep).map(str::to_string).collect()
}

/// `s.toLowerCase()`
pub fn to_lower(s: &str) -> String {
    s.to_lowercase()
}

/// `s.toUpperCase()`
pub fn to_upper(s: &str) -> String {
    s.to_uppercase()
}

/// 템플릿 리터럴: 조각 사이에 식의 `String(x)` 를 넣는다 (`tpl(T3181, &[a, b])` = 원본의 `` `…${a}…${b}…` ``).
/// 조각 수는 식 수 + 1 이어야 한다 (생성된 상수의 주석에 식 원문이 있다)
pub fn tpl(pieces: &[&str], args: &[V]) -> String {
    assert_eq!(pieces.len(), args.len() + 1, "template piece/argument count");
    let mut out = String::from(pieces[0]);
    for (arg, piece) in args.iter().zip(&pieces[1..]) {
        out.push_str(&arg.to_str());
        out.push_str(piece);
    }
    out
}

// ───────────────────────── JSON.stringify ─────────────────────────

fn quote_into(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `JSON.stringify(s)` 의 문자열 따옴표 처리
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    quote_into(&mut out, s);
    out
}

fn stringify_into(v: &V, out: &mut String, stack: &mut Vec<*const ()>) -> Result<bool, Throw> {
    match v {
        V::Undef => Ok(false),
        V::Null => {
            out.push_str("null");
            Ok(true)
        }
        V::Bool(b) => {
            out.push_str(if *b { "true" } else { "false" });
            Ok(true)
        }
        V::Num(n) => {
            if n.is_finite() {
                out.push_str(&num_to_string(*n));
            } else {
                out.push_str("null");
            }
            Ok(true)
        }
        V::Str(s) => {
            quote_into(out, s);
            Ok(true)
        }
        V::Bytes(_) | V::Ext(_) => {
            out.push_str("{}");
            Ok(true)
        }
        V::Arr(a) => {
            let p = Rc::as_ptr(a) as *const ();
            if stack.contains(&p) {
                return Err(Throw::type_error("Converting circular structure to JSON"));
            }
            stack.push(p);
            out.push('[');
            let items = a.borrow().clone();
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if !stringify_into(x, out, stack)? {
                    out.push_str("null");
                }
            }
            out.push(']');
            stack.pop();
            Ok(true)
        }
        V::Obj(o) => {
            let p = Rc::as_ptr(o) as *const ();
            if stack.contains(&p) {
                return Err(Throw::type_error("Converting circular structure to JSON"));
            }
            stack.push(p);
            out.push('{');
            let mut first = true;
            for k in v.keys() {
                let x = o.borrow().get(&k).cloned().unwrap_or_default();
                if x.is_undef() {
                    continue;
                }
                let mark = out.len();
                if !first {
                    out.push(',');
                }
                quote_into(out, &k);
                out.push(':');
                if stringify_into(&x, out, stack)? {
                    first = false;
                } else {
                    out.truncate(mark);
                }
            }
            out.push('}');
            stack.pop();
            Ok(true)
        }
    }
}

/// `JSON.stringify(v)` — undefined 면 None (JS 의 undefined 반환)
pub fn stringify(v: &V) -> Option<String> {
    stringify_checked(v).ok().flatten()
}

/// `JSON.stringify(v)` — 순환 참조면 TypeError
pub fn stringify_checked(v: &V) -> R<Option<String>> {
    let mut out = String::new();
    let wrote = stringify_into(v, &mut out, &mut Vec::new())?;
    Ok(wrote.then_some(out))
}

/// 템플릿에 넣은 `${JSON.stringify(x)}` (undefined 면 "undefined")
pub fn stringify_str(v: &V) -> String {
    stringify(v).unwrap_or_else(|| "undefined".into())
}

// ───────────────────────── JSON.parse ─────────────────────────

/// serde_json 값 → V
pub fn from_json(j: serde_json::Value) -> V {
    match j {
        serde_json::Value::Null => V::Null,
        serde_json::Value::Bool(b) => V::Bool(b),
        serde_json::Value::Number(n) => V::Num(n.as_f64().unwrap_or(f64::NAN)),
        serde_json::Value::String(s) => V::Str(s),
        serde_json::Value::Array(a) => V::from(a.into_iter().map(from_json).collect::<Vec<_>>()),
        serde_json::Value::Object(o) => {
            let out = V::new_obj();
            for (k, v) in o {
                out.put(&k, from_json(v));
            }
            out
        }
    }
}

/// `JSON.parse(s)` — 문자열이 아니면 먼저 String(x) (JS 와 같음)
pub fn parse(v: &V) -> R<V> {
    parse_str(&v.to_str())
}

/// `JSON.parse(text)`
pub fn parse_str(text: &str) -> R<V> {
    // JS 는 앞뒤의 JSON 공백(공백 · 탭 · 줄바꿈 · CR)만 허용한다. serde_json 도 같다.
    serde_json::from_str::<serde_json::Value>(text)
        .map(from_json)
        .map_err(|e| Throw::syntax_error(format!("{e} in JSON")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_print_like_js() {
        for (n, s) in [(1.0, "1"), (0.1 + 0.2, "0.30000000000000004"), (1e21, "1e+21"), (123456789012.0, "123456789012"), (-0.0, "0"), (1.5e-7, "1.5e-7")] {
            assert_eq!(num_to_string(n), s);
        }
        assert_eq!(string_to_number(" 12 "), 12.0);
        assert!(string_to_number("12px").is_nan());
        assert_eq!(string_to_number(""), 0.0);
        assert_eq!(string_to_number("0x1f"), 31.0);
        assert_eq!(parse_int("12px"), 12.0);
        assert_eq!(parse_float("3.5kg"), 3.5);
        assert!(parse_float(".").is_nan());
    }

    #[test]
    fn keys_follow_js_order_and_stringify_drops_undefined() {
        let o = obj! { "b" => 1, "10" => 2, "a" => V::Undef, "2" => "x" };
        assert_eq!(o.keys(), ["2", "10", "b", "a"]);
        assert_eq!(stringify(&o).unwrap(), r#"{"2":"x","10":2,"b":1}"#);
        let a = arr![1, V::Undef, "q\n\"", V::Null];
        assert_eq!(stringify(&a).unwrap(), r#"[1,null,"q\n\"",null]"#);
        assert_eq!(stringify(&V::Undef), None);
        assert_eq!(stringify(&obj! { "d" => V::bytes(vec![1, 2]) }).unwrap(), r#"{"d":{}}"#);
        assert_eq!(stringify(&V::Num(f64::NAN)).unwrap(), "null");
    }

    #[test]
    fn objects_alias_and_reading_undefined_throws() {
        let pages = arr![obj! { "n" => 1 }];
        let page = pages.at(0);
        page.s_("n", V::from(2)).unwrap();
        assert_eq!(pages.at(0).get("n"), V::Num(2.0));
        let copy = pages.deep_clone();
        copy.at(0).put("n", V::from(3));
        assert_eq!(pages.at(0).get("n"), V::Num(2.0));
        let e = V::Undef.g("base").unwrap_err();
        assert_eq!(e.to_string(), "TypeError: Cannot read properties of undefined (reading 'base')");
        assert!(V::Null.s_("x", V::Null).is_err());
        assert_eq!(V::from("abc").s_("x", V::Null).unwrap_err().message, "Cannot create property 'x' on string 'abc'");
        assert_eq!(less_than(&arr![10], &V::from("9")), Some(true), "['10'] < '9' compares strings");
        assert!(V::Undef.push(V::Null).is_err());
    }

    #[test]
    fn loose_equality_and_plus() {
        assert!(loose_eq(&V::from(1), &V::from("1")));
        assert!(loose_eq(&V::Undef, &V::Null));
        assert!(!loose_eq(&V::Null, &V::from(0)));
        assert!(loose_eq(&V::from(true), &V::from("1")));
        assert_eq!(add(&V::from("a"), &V::from(1)), V::from("a1"));
        assert_eq!(add(&V::from(1), &V::from(2)), V::from(3));
        assert_eq!(add(&V::from("x"), &V::Undef), V::from("xundefined"));
        assert_eq!(V::from(vec![V::from(1), V::Null, V::from("a")]).to_str(), "1,,a");
    }

    #[test]
    fn utf16_positions() {
        let s = "가나a😀b";
        assert_eq!(u16_len(s), 6);
        assert_eq!(index_of(s, "b"), 5);
        assert_eq!(substring(s, 1.0, Some(3.0)), "나a");
        assert_eq!(slice(s, -1.0, None), "b");
        assert_eq!(js_trim("\u{FEFF} x \u{3000}"), "x");
    }

    #[test]
    fn parse_keeps_order_and_duplicate_position() {
        let v = parse_str(r#"{"z":1,"a":{"q":[1,2.5,"x"]},"z":3}"#).unwrap();
        assert_eq!(v.keys(), ["z", "a"]);
        assert_eq!(v.get("z"), V::Num(3.0));
        assert_eq!(stringify(&v).unwrap(), r#"{"z":3,"a":{"q":[1,2.5,"x"]}}"#);
        assert_eq!(parse_str("{bad").unwrap_err().name, "SyntaxError");
    }
}
