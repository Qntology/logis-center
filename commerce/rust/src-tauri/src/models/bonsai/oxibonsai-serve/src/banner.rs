//! ASCII art banner and startup messaging for oxibonsai-serve.

/// Crate version pulled at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Full ASCII art banner for OxiBonsai.
pub const BANNER: &str = r#"
  ██████╗ ██╗  ██╗██╗██████╗  ██████╗ ███╗   ██╗███████╗ █████╗ ██╗
 ██╔═══██╗╚██╗██╔╝██║██╔══██╗██╔═══██╗████╗  ██║██╔════╝██╔══██╗██║
 ██║   ██║ ╚███╔╝ ██║██████╔╝██║   ██║██╔██╗ ██║███████╗███████║██║
 ██║   ██║ ██╔██╗ ██║██╔══██╗██║   ██║██║╚██╗██║╚════██║██╔══██║██║
 ╚██████╔╝██╔╝ ██╗██║██████╔╝╚██████╔╝██║ ╚████║███████║██║  ██║██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝
"#;

/// Print the banner to stderr.
pub fn print_banner() {
    eprintln!("{}", BANNER);
}

/// Build a human-readable startup message showing the listening address.
///
/// # Examples
///
/// ```
/// use oxibonsai_serve::banner::startup_message;
/// let msg = startup_message("0.0.0.0", 8080);
/// assert!(msg.contains("0.0.0.0:8080"));
/// ```
pub fn startup_message(host: &str, port: u16) -> String {
    format!(
        "OxiBonsai v{VERSION} listening on http://{host}:{port}\n\
         Endpoints:\n\
         \x20 POST /v1/chat/completions\n\
         \x20 GET  /v1/models\n\
         \x20 GET  /health\n\
         \x20 GET  /metrics"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_is_nonempty() {
        assert!(!BANNER.is_empty());
    }

    #[test]
    fn version_is_nonempty() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn startup_message_contains_addr() {
        let msg = startup_message("127.0.0.1", 3000);
        assert!(msg.contains("127.0.0.1:3000"));
    }

    #[test]
    fn startup_message_lists_endpoints() {
        let msg = startup_message("0.0.0.0", 8080);
        assert!(msg.contains("/v1/chat/completions"));
        assert!(msg.contains("/v1/models"));
        assert!(msg.contains("/health"));
        assert!(msg.contains("/metrics"));
    }
}
