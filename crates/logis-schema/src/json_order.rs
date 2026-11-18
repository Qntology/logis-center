//! JSON 객체 키 순서를 예전(serde_json 기본 = BTreeMap, 키 정렬)과 같게 지키는 도우미.
//!
//! lancedb 0.41 의 매니페스트가 `serde_json` 의 `preserve_order` 기능을 켜고, Cargo 기능은 합쳐지므로 이 앱의
//! `serde_json::Map` 은 IndexMap(넣은 순서 · 문서 순서)이 됩니다. 0.31 까지는 늘 키 정렬 순서였고, 색인 문장 · 프롬프트 ·
//! 저장 문자열 · 화면 순서가 그 순서를 따랐습니다. 순서가 출력에 드러나는 곳(맵 순회 · 직렬화)은 여기 도우미를 써서
//! 정렬 순서를 지킵니다. `preserve_order` 가 꺼져 있으면 맵이 이미 정렬돼 있어 결과가 같고, 정렬 비용만 듭니다.
//!
//! - 순회: [`sorted_iter`] · [`sorted_keys`] · [`sorted_values`] · [`sorted_iter_mut`] · [`sorted_values_mut`] · [`into_sorted`]
//! - 들어온 값을 한 번에: [`sort_all`] (파싱 · `to_value` 직후, 하위 객체까지)
//! - 직렬화: [`to_string_sorted`] · [`to_string_pretty_sorted`] · [`to_vec_sorted`], 형식 문자열에는 [`Sorted`]
//!
//! 키 비교는 `String` 의 `Ord`(바이트 사전 순)로, BTreeMap<String, Value> 의 순서와 같습니다.

use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde_json::{Map, Value};

/// (키, 값) 을 키 정렬 순서로 — `map.iter()` 대신.
pub fn sorted_iter(map: &Map<String, Value>) -> std::vec::IntoIter<(&String, &Value)> {
    let mut v: Vec<(&String, &Value)> = map.iter().collect();
    v.sort_unstable_by(|a, b| a.0.cmp(b.0));
    v.into_iter()
}

/// 키를 정렬 순서로 — `map.keys()` 대신.
pub fn sorted_keys(map: &Map<String, Value>) -> std::vec::IntoIter<&String> {
    let mut v: Vec<&String> = map.keys().collect();
    v.sort_unstable();
    v.into_iter()
}

/// 값을 키 정렬 순서로 — `map.values()` 대신.
pub fn sorted_values(map: &Map<String, Value>) -> std::vec::IntoIter<&Value> {
    sorted_iter(map).map(|(_, v)| v).collect::<Vec<_>>().into_iter()
}

/// (키, 바꿀 수 있는 값) 을 키 정렬 순서로 — `map.iter_mut()` 대신.
pub fn sorted_iter_mut(map: &mut Map<String, Value>) -> std::vec::IntoIter<(&String, &mut Value)> {
    let mut v: Vec<(&String, &mut Value)> = map.iter_mut().collect();
    v.sort_unstable_by(|a, b| a.0.cmp(b.0));
    v.into_iter()
}

/// 바꿀 수 있는 값을 키 정렬 순서로 — `map.values_mut()` 대신.
pub fn sorted_values_mut(map: &mut Map<String, Value>) -> std::vec::IntoIter<&mut Value> {
    sorted_iter_mut(map).map(|(_, v)| v).collect::<Vec<_>>().into_iter()
}

/// 맵을 소비하며 (키, 값) 을 키 정렬 순서로 — `map.into_iter()` · `for (k, v) in map` 대신.
pub fn into_sorted(map: Map<String, Value>) -> std::vec::IntoIter<(String, Value)> {
    let mut v: Vec<(String, Value)> = map.into_iter().collect();
    v.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    v.into_iter()
}

/// 값 안의 모든 객체(하위 포함)를 키 정렬 순서로 다시 놓습니다. 파싱 · `serde_json::to_value` 직후에 부르면 그 뒤 순회 ·
/// 직렬화가 예전 순서입니다 (그 값에 새 키를 넣으면 뒤에 붙으므로, 순서가 드러나는 곳은 위 순회 도우미를 씁니다).
pub fn sort_all(v: &mut Value) {
    v.sort_all_objects();
}

/// [`sort_all`] 을 한 값으로.
pub fn sorted(mut v: Value) -> Value {
    v.sort_all_objects();
    v
}

/// 키 정렬 순서로 직렬화하는 감싸개. `serde_json::to_string(&Sorted(&v))` 나 `format!("{}", Sorted(&v))` 는
/// 예전 `serde_json::to_string(&v)` · `v.to_string()` 과 같은 문자열입니다(값을 복사하지 않음).
#[derive(Clone, Copy)]
pub struct Sorted<'a>(pub &'a Value);

impl Serialize for Sorted<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Value::Object(map) => {
                let entries: Vec<(&String, &Value)> = sorted_iter(map).collect();
                let mut out = serializer.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    out.serialize_entry(k, &Sorted(v))?;
                }
                out.end()
            }
            Value::Array(items) => {
                let mut out = serializer.serialize_seq(Some(items.len()))?;
                for v in items {
                    out.serialize_element(&Sorted(v))?;
                }
                out.end()
            }
            other => other.serialize(serializer),
        }
    }
}

/// `Map` 을 키 정렬 순서로 직렬화하는 감싸개 — `serde_json::to_string(&m)` 대신 `serde_json::to_string(&SortedMap(&m))`.
#[derive(Clone, Copy)]
pub struct SortedMap<'a>(pub &'a Map<String, Value>);

impl Serialize for SortedMap<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let entries: Vec<(&String, &Value)> = sorted_iter(self.0).collect();
        let mut out = serializer.serialize_map(Some(entries.len()))?;
        for (k, v) in entries {
            out.serialize_entry(k, &Sorted(v))?;
        }
        out.end()
    }
}

/// 값 목록(`Vec<Value>` · `&[Value]`)을 원소마다 키 정렬 순서로 직렬화하는 감싸개 — `SortedSlice(&values)`.
#[derive(Clone, Copy)]
pub struct SortedSlice<'a>(pub &'a [Value]);

impl Serialize for SortedSlice<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut out = serializer.serialize_seq(Some(self.0.len()))?;
        for v in self.0 {
            out.serialize_element(&Sorted(v))?;
        }
        out.end()
    }
}

impl std::fmt::Display for Sorted<'_> {
    /// `Value` 의 `Display` 처럼: 기본은 한 줄, `{:#}` 은 들여쓰기.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = if f.alternate() { to_string_pretty_sorted(self.0) } else { to_string_sorted(self.0) };
        f.write_str(&s)
    }
}

/// `serde_json::to_string(v)` 를 키 정렬 순서로. Value 직렬화는 실패하지 않습니다.
pub fn to_string_sorted(v: &Value) -> String {
    serde_json::to_string(&Sorted(v)).unwrap_or_default()
}

/// `serde_json::to_string_pretty(v)` 를 키 정렬 순서로.
pub fn to_string_pretty_sorted(v: &Value) -> String {
    serde_json::to_string_pretty(&Sorted(v)).unwrap_or_default()
}

/// `serde_json::to_vec(v)` 를 키 정렬 순서로.
pub fn to_vec_sorted(v: &Value) -> Vec<u8> {
    serde_json::to_vec(&Sorted(v)).unwrap_or_default()
}

/// 구조체의 `Value` 필드를 키 정렬 순서로 직렬화합니다: `#[serde(serialize_with = "logis_schema::json_order::serialize_sorted")]`.
pub fn serialize_sorted<S: Serializer>(v: &Value, serializer: S) -> Result<S::Ok, S::Error> {
    Sorted(v).serialize(serializer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shuffled() -> Value {
        // preserve_order 이면 넣은 순서가 남는 값 (b, a, c / 안쪽 z, y)
        let mut inner = Map::new();
        inner.insert("z".into(), json!(1));
        inner.insert("y".into(), json!([{"q": 1, "p": 2}]));
        let mut m = Map::new();
        m.insert("b".into(), json!(2));
        m.insert("a".into(), Value::Object(inner));
        m.insert("c".into(), json!(null));
        Value::Object(m)
    }

    #[test]
    fn sorted_serialization_matches_btreemap_order() {
        let v = shuffled();
        let want = r#"{"a":{"y":[{"p":2,"q":1}],"z":1},"b":2,"c":null}"#;
        assert_eq!(to_string_sorted(&v), want);
        assert_eq!(format!("{}", Sorted(&v)), want);
        assert_eq!(to_string_sorted(&sorted(v.clone())), want);
        assert_eq!(serde_json::to_string(&sorted(v.clone())).unwrap(), want);
        assert_eq!(to_string_pretty_sorted(&v), serde_json::to_string_pretty(&sorted(v)).unwrap());
    }

    #[test]
    fn sorted_iteration_orders_keys() {
        let mut v = shuffled();
        let m = v.as_object().unwrap();
        assert_eq!(sorted_keys(m).map(|k| k.as_str()).collect::<Vec<_>>(), vec!["a", "b", "c"]);
        assert_eq!(sorted_iter(m).map(|(k, _)| k.as_str()).collect::<Vec<_>>(), vec!["a", "b", "c"]);
        assert_eq!(sorted_values(m).next(), m.get("a"));
        let m = v.as_object_mut().unwrap();
        assert_eq!(sorted_iter_mut(m).map(|(k, _)| k.clone()).collect::<Vec<_>>(), vec!["a", "b", "c"]);
        for x in sorted_values_mut(m) {
            *x = json!(0);
        }
        assert_eq!(to_string_sorted(&v), r#"{"a":0,"b":0,"c":0}"#);
        let owned = v.as_object().unwrap().clone();
        assert_eq!(into_sorted(owned).map(|(k, _)| k).collect::<Vec<_>>(), vec!["a", "b", "c"]);
    }
}
