//! topo-benchmark CLI entry point.

use anyhow::{Context, Result};
use clap::Parser;

use topo_benchmark::cli::{BenchCommand, Cli};
use topo_benchmark::types::Dimension;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        BenchCommand::Run(args) => cmd_run(args),
        BenchCommand::Compare(args) => cmd_compare(args),
        BenchCommand::Report(args) => cmd_report(args),
    }
}

fn cmd_run(args: topo_benchmark::cli::RunArgs) -> Result<()> {
    let dimensions: Vec<Dimension> = if args.dimension == "all" {
        Dimension::all().to_vec()
    } else {
        let d = Dimension::from_str(&args.dimension)
            .with_context(|| format!("unknown dimension: {}", args.dimension))?;
        vec![d]
    };

    let dataset_root = args
        .dataset_root
        .unwrap_or_else(|| topo_benchmark::datasets::default_dataset_root().unwrap());

    let output_dir = args.output_dir.unwrap_or_else(|| {
        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
        std::path::PathBuf::from(format!(".benchmark/runs/{ts}"))
    });

    let run = topo_benchmark::runner::run_benchmark(
        &dimensions,
        &args.split,
        &dataset_root,
        &args.edge_kind,
    )?;

    topo_benchmark::runner::write_artifacts(&run, &output_dir)?;

    // Print scorecard summary.
    println!("Benchmark run complete → {}", output_dir.display());
    println!(
        "Overall: {:.4}  |  Decision: {}",
        run.scorecard.overall_primary, run.scorecard.promotion_decision
    );
    for dim in Dimension::all() {
        if let Some(score) = run.scorecard.dimensions.get(dim.as_str()) {
            println!("  {}: {:.4}", dim.as_str(), score);
        }
    }
    if !run.scorecard.failing_cases.is_empty() {
        println!(
            "Failing cases: {}",
            run.scorecard.failing_cases.join(", ")
        );
    }

    Ok(())
}

fn cmd_compare(args: topo_benchmark::cli::CompareArgs) -> Result<()> {
    let result = topo_benchmark::compare::compare_runs(
        &args.candidate,
        &args.reference,
        args.max_regression,
    )?;

    println!(
        "Comparison: {} → {}",
        args.reference.display(),
        args.candidate.display()
    );
    println!(
        "Overall: {:.4} → {:.4}  (Δ {:.4})",
        result.reference_overall, result.candidate_overall, result.overall_delta
    );
    for (name, delta) in &result.dimensions {
        let marker = if delta.regressed { " ← REGRESSION" } else { "" };
        println!(
            "  {name}: {:.4} → {:.4}  (Δ {:.4}){marker}",
            delta.reference, delta.candidate, delta.delta
        );
    }
    println!("Decision: {}", result.promotion_decision);

    if args.fail_on_regression && result.promotion_decision == "fail" {
        std::process::exit(1);
    }

    Ok(())
}

fn cmd_report(args: topo_benchmark::cli::ReportArgs) -> Result<()> {
    let scorecard = topo_benchmark::scorecard::load_scorecard(&args.input.join("scorecard.json"))?;
    let report = topo_benchmark::report::generate_summary(&scorecard, &Default::default(), &Default::default(), None);
    println!("{report}");
    Ok(())
}
