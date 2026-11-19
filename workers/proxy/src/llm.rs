//! `Deepinfra(key, model, system, user, config, inlineData)` · `Gemini(...)` — 원본 그대로 (같은 요청 본문, 같은 응답 해석).

use logis_worker_common::dirty_json::parse_dirty_json;
use logis_worker_common::js::{self, Body};
use logis_worker_common::jsv::{add, index_of, js_trim, parse_str, stringify_checked, V};
use logis_worker_common::{console_log, obj, R};

/// 응답 글에서 JSON 을 꺼내는 원본의 세 단계 (그대로 → ``` 걷어 내고 → 더러운 JSON 고치기). 모두 실패하면 글 그대로
fn salvage(content: String) -> V {
    if let Ok(v) = parse_str(&content) {
        return v;
    }
    let mut content = content;
    if index_of(&content, "```") > -1 {
        content = strip_fences(&content);
        if let Ok(v) = parse_str(&content) {
            return v;
        }
    }
    let cleaned = strip_fences(&content);
    // 원본: content = parseDirtyJson(content); return JSON.parse(content) — 실패하면(객체는 "[object Object]") catch 뒤 content(고친 값).
    // 그래서 null 은 null, 객체는 그 객체, 수 하나짜리 배열 [5] 는 5 가 된다.
    let fixed = parse_dirty_json(&V::from(cleaned));
    match parse_str(&fixed.to_str()) {
        Ok(v) => v,
        Err(_) => fixed,
    }
}

/// replace(/```json/gi,"") · replace(/```/gi,"") · replace(/\n/gi,"") · trim()
fn strip_fences(s: &str) -> String {
    let lower_json = remove_ci(s, "```json");
    let no_fence = lower_json.replace("```", "");
    js_trim(&no_fence.replace('\n', "")).to_string()
}

/// 대소문자를 가리지 않고 needle 을 모두 지운다 (needle 은 ASCII)
fn remove_ci(s: &str, needle: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let (hay, n) = (s.as_bytes(), needle.as_bytes());
    let mut i = 0;
    while i < hay.len() {
        if i + n.len() <= hay.len() && hay[i..i + n.len()].eq_ignore_ascii_case(n) {
            i += n.len();
        } else {
            let ch_len = s[i..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&s[i..i + ch_len]);
            i += ch_len;
        }
    }
    out
}

/// `Deepinfra(key, model, system, user, config, inlineData)`
pub async fn deepinfra(key: &V, model: &V, system: &V, user: &V, config: &V, inline_data: &V) -> R<V> {
    let messages = V::new_arr();
    if system.truthy() {
        messages.push(obj! { "role" => "system", "content" => system.clone() })?;
    }
    if inline_data.truthy() {
        messages.push(obj! {
            "role" => "user",
            "content" => vec![
                obj! { "type" => "text", "text" => add(system, user) },
                obj! { "type" => "image_url", "image_url" => obj! { "url" => inline_data.g("data")? } },
            ],
        })?;
        console_log!("inlineData.data", inline_data.g("data")?.length()?);
    } else if user.truthy() {
        messages.push(obj! { "role" => "user", "content" => user.clone() })?;
    }

    let temperature = if config.truthy() { config.g("temperature")? } else { V::Num(0.95) };
    let mut body = obj! {
        "model" => model.clone(),
        "messages" => messages,
        "max_tokens" => 15000,
        "temperature" => temperature,
        "top_p" => 1,
    };

    let mut pathname = "chat/completions";
    let model_str = match model {
        V::Str(s) => s.clone(),
        _ => return Err(logis_worker_common::Throw::type_error("model.indexOf is not a function")),
    };
    let is_embedding = index_of(&model_str, "google/embeddinggemma-300m") > -1;
    if is_embedding {
        pathname = "embeddings";
        body = obj! { "input" => add(system, user), "model" => model.clone(), "encoding_format" => "float" };
    }

    let auth = format!("Bearer {}", key.to_str());
    let text = stringify_checked(&body)?.unwrap_or_default();
    let res = js::fetch(
        &format!("https://api.deepinfra.com/v1/openai/{pathname}"),
        "POST",
        &[("Authorization", &auth), ("Content-Type", "application/json")],
        Body::Text(text),
    )
    .await?;
    let json = js::response_json(&res).await?;

    if is_embedding {
        return json.g("data")?.gi(0)?.g("embedding");
    }
    let content = json.g("choices")?.gi(0)?.g("message")?.g("content")?;
    let content = match content {
        V::Str(s) => js_trim(&s).to_string(),
        other => return Err(logis_worker_common::Throw::type_error(format!("content.trim is not a function ({})", other.type_of()))),
    };
    console_log!("content", content.clone());
    if config.truthy() {
        return Ok(V::Str(content));
    }
    Ok(salvage(content))
}

/// `Gemini(key, model, system, user, config, inlineData)`
pub async fn gemini(key: &V, model: &V, system: &V, user: &V, config: &V, inline_data: &V) -> R<V> {
    console_log!("Gemini 진입");
    let config = if config.is_undef() {
        obj! { "response_mime_type" => "application/json", "temperature" => 1 }
    } else {
        config.clone()
    };
    let parts = V::from(vec![obj! { "text" => add(system, user) }]);
    if inline_data.truthy() {
        parts.push(obj! { "inlineData" => inline_data.clone() })?;
    }
    let body = obj! {
        "contents" => vec![obj! { "parts" => parts }],
        "generationConfig" => config.clone(),
    };
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model.to_str(),
        key.to_str()
    );
    let res = js::fetch(&url, "POST", &[("Content-Type", "application/json")], Body::Text(stringify_checked(&body)?.unwrap_or_default())).await?;
    let data = js::response_json(&res).await?;
    let content = data.g("candidates")?.gi(0)?.g("content")?.g("parts")?.gi(0)?.g("text")?;

    if config.g("response_mime_type")?.truthy() {
        let text = content.to_str();
        return Ok(match parse_str(&text) {
            Ok(v) => v,
            Err(_) => salvage_gemini(text),
        });
    }
    Ok(content)
}

/// Gemini 의 두 번째 · 세 번째 단계 (원본은 첫 단계를 따로 했다 — 결과는 [`salvage`] 와 같다)
fn salvage_gemini(text: String) -> V {
    salvage(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salvages_like_the_original() {
        assert_eq!(salvage(r#"{"a":1}"#.into()).get("a"), V::Num(1.0));
        let fenced = "```JSON\n{\"a\": 2}\n```".to_string();
        assert_eq!(salvage(fenced).get("a"), V::Num(2.0));
        assert_eq!(salvage("{a: 'x',}".into()).get("a"), V::from("x"));
        assert_eq!(salvage("not json".into()), V::Null);
    }
}
