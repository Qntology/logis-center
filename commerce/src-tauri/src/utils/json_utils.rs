use serde_json::Value;

pub fn merge_node(obj1: &Value, obj2: &Value) -> Value {
    let mut merged = obj1.clone();
    if let (Some(m_obj), Some(o2_obj)) = (merged.as_object_mut(), obj2.as_object()) {
        for (k, v) in o2_obj {
            let is_empty = match v {
                Value::Null => true,
                Value::String(s) => s.is_empty(),
                Value::Number(n) => n.as_f64().unwrap_or(0.0) == 0.0,
                _ => false,
            };
            if !is_empty {
                m_obj.insert(k.clone(), v.clone());
            }
        }
    }
    merged
}