//! `renderHtml(cookies, rows)` — 원본 src/renderHtml.ts.
//!
//! 템플릿 리터럴 하나뿐이고 `${}` 식이 없다(두 인자는 쓰지 않음). 원본 파일은 CRLF 이지만 템플릿 리터럴은 CR LF 를 LF 로 바꾸므로
//! (ECMA-262 의 TV 규칙) 줄 끝은 `\n` 이다. 앞의 줄바꿈과 끝의 `\n\t` 까지 바이트 그대로.

use logis_worker_common::V;

const HTML: &str = "\n\t\t<!DOCTYPE html>\n\t\t<html lang=\"en\">\n\t\t\t<head>\n\t\t\t\t<meta charset=\"UTF-8\" />\n\t\t\t\t<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,minimum-scale=1,maximum-scale=1,user-scalable=no\">\n\t\t\t\t<title>Logis.center</title>\n\t\t\t</head>\n\t\t\t<body>\n\t\t\t\t<script src=\"/static/js/content.js\"></script>\n\t\t\t</body>\n\t\t</html>\n\t";

pub fn render_html(_cookies: &V, _list: &V) -> String {
    HTML.to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn same_bytes_as_the_template_literal() {
        let html = super::render_html(&super::V::Undef, &super::V::Undef);
        assert!(html.starts_with("\n\t\t<!DOCTYPE html>\n"));
        assert!(html.ends_with("</html>\n\t"));
        assert!(!html.contains('\r'));
    }
}
