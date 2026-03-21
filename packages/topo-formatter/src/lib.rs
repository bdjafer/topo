//! Text formatter for topo structural analysis output.
//!
//! Consumes analysis.schema.json dicts and produces human/LLM-readable text.

pub mod context;
pub mod dag;
pub mod domain;
pub mod format;
pub mod style;

#[cfg(feature = "python")]
mod python;

pub use context::format_context;
pub use format::format_text;
