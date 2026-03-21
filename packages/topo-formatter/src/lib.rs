//! Text formatter for topo structural analysis output.
//!
//! Consumes analysis.schema.json dicts and produces human/LLM-readable text.

pub mod dag;
pub mod format;
pub mod style;

#[cfg(feature = "python")]
mod python;

pub use format::format_text;
