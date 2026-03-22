//! Text formatters for topo structural analysis output.
//!
//! Three capability-focused formatters plus domain model and LLM context.

pub mod context;
pub mod dag;
pub mod domain;
pub mod format;
pub mod style;

#[cfg(feature = "python")]
mod python;

pub use context::format_context;
pub use format::{format_health, format_issues};
