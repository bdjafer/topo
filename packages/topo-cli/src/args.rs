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
    pub command: Command,

    /// Output JSON instead of text.
    #[arg(long, global = true)]
    pub json: bool,

    /// Disable colored output.
    #[arg(long = "no-color", global = true)]
    pub no_color: bool,

    /// Skip parse cache (force re-parse).
    #[arg(long = "no-cache", global = true)]
    pub no_cache: bool,

    /// Path to pre-computed embeddings JSON file.
    #[arg(long, global = true)]
    pub embeddings: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Structural analysis — issues + health metrics.
    Analyze(AnalyzeArgs),
    /// Architecture and domain decomposition (not yet implemented).
    Domain(DomainArgs),
    /// Web UI (not yet implemented).
    Serve(ServeArgs),
    /// Parse a project into CodeGraph JSON.
    Parse(ParseArgs),
    /// Manage the parse cache.
    Cache(CacheArgs),
    /// Analyze a pre-parsed graph.json file (benchmark tooling).
    AnalyzeRaw(AnalyzeRawArgs),
    /// Apply a mutation to a pre-parsed graph (benchmark tooling).
    Mutate(MutateArgs),
    /// Export R-GIN training features (spectral PE, RWPE, tree features, etc.).
    ExportFeatures(ExportFeaturesArgs),
}

#[derive(Parser)]
pub struct AnalyzeArgs {
    /// Path to project root.
    pub path: PathBuf,

    /// Minimum issue severity to report: high, medium, low.
    #[arg(long, value_parser = ["high", "medium", "low"])]
    pub severity: Option<String>,

    /// Track health over git history instead of current snapshot.
    #[arg(long)]
    pub history: bool,

    /// Only consider commits after this date (YYYY-MM-DD). Implies --history.
    #[arg(long)]
    pub since: Option<String>,

    /// Sampling strategy (with --history): weekly, monthly.
    #[arg(long, default_value = "weekly")]
    pub sample: String,

    /// Maximum number of commits to analyze (with --history).
    #[arg(long, default_value = "20")]
    pub max_commits: usize,
}

#[derive(Parser)]
pub struct DomainArgs {
    /// Path to project root.
    pub path: PathBuf,
}

#[derive(Parser)]
pub struct ServeArgs {
    /// Path to project root.
    pub path: PathBuf,

    /// Port to listen on.
    #[arg(long, default_value = "8080")]
    pub port: u16,
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

    /// Source language (auto-detected if omitted).
    #[arg(long, value_parser = ["rust", "python"])]
    pub language: Option<String>,
}

#[derive(Parser)]
pub struct CacheArgs {
    #[command(subcommand)]
    pub command: CacheCommand,
}

#[derive(Parser)]
pub struct AnalyzeRawArgs {
    /// Path to pre-parsed graph.json file.
    #[arg(long)]
    pub input: PathBuf,
}

#[derive(Parser)]
pub struct MutateArgs {
    /// Path to pre-parsed graph.json file.
    #[arg(long)]
    pub input: PathBuf,

    /// Mutation type: inject_cycle, layer_violation, overloaded_utility, wide_interface, near_disconnect.
    #[arg(long = "type", value_parser = ["inject_cycle", "layer_violation", "overloaded_utility", "wide_interface", "near_disconnect"])]
    pub mutation_type: String,

    /// Severity level (1 = mild, 2 = medium, 3 = severe).
    #[arg(long, default_value = "2")]
    pub severity: u8,

    /// Deterministic seed for random targeting.
    #[arg(long, default_value = "42")]
    pub seed: u64,
}

#[derive(Parser)]
pub struct ExportFeaturesArgs {
    /// Path to project root (mutually exclusive with --input).
    pub path: Option<PathBuf>,

    /// Pre-parsed graph JSON file (skips parsing).
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Output file path (.npz). A .meta.json sidecar is written alongside.
    #[arg(long, short)]
    pub output: PathBuf,
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
