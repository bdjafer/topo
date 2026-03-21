//! CLI argument definitions.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "topo-benchmark",
    about = "Benchmark harness for topo-analyzer",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: BenchCommand,
}

#[derive(Subcommand)]
pub enum BenchCommand {
    /// Run benchmark evaluation.
    Run(RunArgs),
    /// Compare two benchmark runs.
    Compare(CompareArgs),
    /// Regenerate a summary report from an existing run.
    Report(ReportArgs),
}

#[derive(Parser)]
pub struct RunArgs {
    /// Which dimension(s) to run: all, architecture, mutations, stability, anomalies.
    #[arg(long, default_value = "all")]
    pub dimension: String,

    /// Dataset split to evaluate.
    #[arg(long, default_value = "public")]
    pub split: String,

    /// Path to benchmark/datasets/ directory.
    #[arg(long)]
    pub dataset_root: Option<PathBuf>,

    /// Output directory for run artifacts.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Edge kind(s) for analysis. "combined" uses calls+imports+inherits.
    #[arg(long, default_value = "combined")]
    pub edge_kind: String,
}

#[derive(Parser)]
pub struct CompareArgs {
    /// Path to candidate run directory.
    pub candidate: PathBuf,

    /// Path to reference run directory.
    pub reference: PathBuf,

    /// Maximum allowed dimension regression (absolute).
    #[arg(long, default_value = "0.03")]
    pub max_regression: f64,

    /// Exit with code 1 if promotion decision is "fail".
    #[arg(long)]
    pub fail_on_regression: bool,
}

#[derive(Parser)]
pub struct ReportArgs {
    /// Path to a benchmark run directory.
    pub input: PathBuf,
}
