//! CLI argument definitions using clap derive.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "topo",
    about = "Structural intelligence for codebases",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Path to project root (default mode: parse + analyze).
    #[arg(global = false)]
    pub path: Option<PathBuf>,

    #[command(flatten)]
    pub analysis: AnalysisArgs,

    /// Comma-separated directory names to exclude.
    #[arg(long)]
    pub exclude: Option<String>,

    /// Analysis scope preset for monorepos.
    #[arg(long, value_parser = ["auto", "all", "first-party"])]
    pub scope: Option<String>,

    /// Source language (auto-detected if omitted).
    #[arg(long, value_parser = ["rust", "python"])]
    pub language: Option<String>,

    /// Disable parse cache (force re-parse).
    #[arg(long)]
    pub no_cache: bool,
}

#[derive(Subcommand)]
pub enum Command {
    /// Parse a project into CodeGraph JSON.
    Parse(ParseArgs),
    /// Analyze a codebase (auto-parses if no --input given).
    Analyze(AnalyzeArgs),
    /// Manage the parse cache.
    Cache(CacheArgs),
}

#[derive(Parser)]
pub struct ParseArgs {
    /// Path to project root.
    pub path: PathBuf,

    /// Output file (default: stdout).
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Comma-separated directory names to exclude.
    #[arg(long)]
    pub exclude: Option<String>,

    /// Analysis scope preset for monorepos.
    #[arg(long, value_parser = ["auto", "all", "first-party"])]
    pub scope: Option<String>,

    /// Source language (auto-detected if omitted).
    #[arg(long, value_parser = ["rust", "python"])]
    pub language: Option<String>,
}

#[derive(Parser)]
pub struct AnalyzeArgs {
    /// Path to project root (will auto-parse).
    pub path: Option<PathBuf>,

    /// Path to pre-parsed CodeGraph JSON file (skips parsing).
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Comma-separated directory names to exclude (used with path, not --input).
    #[arg(long)]
    pub exclude: Option<String>,

    /// Analysis scope preset for monorepos (used with path, not --input).
    #[arg(long, value_parser = ["auto", "all", "first-party"])]
    pub scope: Option<String>,

    /// Source language (auto-detected if omitted).
    #[arg(long, value_parser = ["rust", "python"])]
    pub language: Option<String>,

    #[command(flatten)]
    pub analysis: AnalysisArgs,
}

#[derive(Parser, Clone)]
pub struct AnalysisArgs {
    /// Output as JSON.
    #[arg(long = "json")]
    pub as_json: bool,

    /// Edge layer to analyze (calls, imports, inherits, defines, combined).
    #[arg(long = "edge-kind", default_value = "combined")]
    pub edge_kind: String,

    /// Number of modules (auto if omitted).
    #[arg(long = "n-modules")]
    pub n_modules: Option<usize>,

    /// Analysis level (package, module, symbol).
    #[arg(long, value_parser = ["package", "module", "symbol"])]
    pub level: Option<String>,

    /// Show full details.
    #[arg(short, long)]
    pub verbose: bool,

    /// Show spectral diagnostics.
    #[arg(long)]
    pub diagnostics: bool,

    /// Disable colored output.
    #[arg(long = "no-color")]
    pub no_color: bool,
}

#[derive(Parser)]
pub struct CacheArgs {
    #[command(subcommand)]
    pub command: CacheCommand,
}

#[derive(Subcommand)]
pub enum CacheCommand {
    /// Remove cached parse results.
    Clear {
        /// Path to project root.
        #[arg(default_value = ".")]
        path: PathBuf,
    },
}
