//! Terminal styling with compile-time feature gating.
//!
//! When the `terminal` feature is enabled, uses `owo-colors` for ANSI codes.
//! Without it, all methods return plain text.

#[cfg(feature = "terminal")]
use owo_colors::OwoColorize;

pub struct Style {
    enabled: bool,
}

impl Style {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn bold(&self, text: &str) -> String {
        #[cfg(feature = "terminal")]
        if self.enabled {
            return format!("{}", text.bold());
        }
        text.to_string()
    }

    pub fn dim(&self, text: &str) -> String {
        #[cfg(feature = "terminal")]
        if self.enabled {
            return format!("{}", text.dimmed());
        }
        text.to_string()
    }

    pub fn red_text(&self, text: &str) -> String {
        #[cfg(feature = "terminal")]
        if self.enabled {
            return format!("{}", text.red());
        }
        text.to_string()
    }

    pub fn yellow_text(&self, text: &str) -> String {
        #[cfg(feature = "terminal")]
        if self.enabled {
            return format!("{}", text.yellow());
        }
        text.to_string()
    }

    pub fn cyan(&self, text: &str) -> String {
        #[cfg(feature = "terminal")]
        if self.enabled {
            return format!("{}", text.cyan());
        }
        text.to_string()
    }

    pub fn green(&self, text: &str) -> String {
        #[cfg(feature = "terminal")]
        if self.enabled {
            return format!("{}", text.green());
        }
        text.to_string()
    }

    pub fn severity(&self, label: &str, text: &str) -> String {
        match label {
            "high" => self.red_text(text),
            "medium" => self.yellow_text(text),
            _ => self.dim(text),
        }
    }
}
