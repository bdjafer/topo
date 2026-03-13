# Topo — Structural Intelligence for Codebases

## Project Structure

Monorepo with three packages under `packages/`:

- **topo-parser**: Source code → typed multilayer graph. Parses Python codebases into nodes (functions, classes, modules) and edges (calls, imports, inheritance, co-location).
- **topo-analyzer**: Graph → structural intelligence. Spectral decomposition, module detection, role classification, anomaly detection.
- **topo-cli**: Developer-facing interface. CLI commands that run the pipeline and produce human/LLM-readable output.

## Development

```bash
uv sync                    # Install all packages in dev mode
uv run pytest              # Run all tests
uv run topo <path>         # Run the CLI
```

## Design Decisions

### Clustering quality metric: NMI, not ARI

We use **Normalized Mutual Information (NMI)** to compare spectral modules against directory-based baselines. We deliberately do not use **Adjusted Rand Index (ARI)**.

ARI counts pairs of items that are co-clustered in both partitions. This makes it degenerate or misleading for our use case:

- **At module level**, the file-based baseline produces singleton clusters (each module-level node is its own file), so ARI is mathematically 0 regardless of clustering quality.
- **At any level**, the spectral clustering intentionally produces finer-grained modules than directory structure — splitting a flat package like `flask.*` into 5 sub-groups based on coupling. ARI penalizes this refinement as "disagreement," even when the sub-groups are architecturally correct.
- **Coarsening the baseline** to package level still yields low ARI (~0.06) because the baseline is extremely imbalanced (18/3/3 for Flask).

NMI uses entropy rather than pair-counting, so it correctly measures how much knowing the spectral cluster tells you about the file-module identity, even when the partitions have different granularities. NMI of 0.7–0.8 means the spectral modules are strongly consistent with file structure while revealing finer internal structure — which is the tool's purpose.

---

# Structural Intelligence

## Permanent Context

---

### What We Know

Every codebase has two descriptions. The first is the source code itself — the text that compilers and humans read. The second is the topology — the graph of which entities depend on, call, import, inherit from, and co-evolve with each other. These are different descriptions of the same reality.

Current tooling operates almost entirely on the first description. Linters check syntax. Type checkers verify contracts. LLMs read text. Dependency analyzers list edges. All of these work on local properties — what a single entity is, what it directly touches.

The topology contains global structural information that no local analysis can access. Which regions of the codebase are tightly coupled. Where the natural architectural boundaries actually are (as opposed to where the directory structure claims they are). Which entities play structurally critical roles — bridges between subsystems, hubs that everything depends on, orphans that nothing reaches. How the structural reality drifts over time as the codebase evolves.

This structural information exists in the graph. Developers currently extract it through experience, intuition, and expensive manual reasoning. No tool extracts it systematically.

Spectral graph analysis — decomposing a graph into its natural resonance modes via eigendecomposition of its adjacency or Laplacian matrix — is the established mathematical method for extracting global structural properties from graphs. It is well-studied, computationally tractable, and available in standard numerical libraries. It has been successfully applied to social networks, biological networks, and infrastructure networks. It has not been applied to code structure as a developer-facing tool.

---

### What We Want

A tool that takes a codebase as input and produces structural intelligence as output.

Structural intelligence means: answers to questions about the global topology that cannot be answered by reading individual files or tracing individual paths. Questions like:

- What are the actual architectural modules of this codebase, derived from how the code is actually connected rather than how the directories are organized?
- Which entities are structurally critical — positioned such that their modification has outsized impact on the rest of the system?
- Which entities are structurally anomalous — occupying positions in the graph that are inconsistent with their apparent purpose?
- Is the architecture stable, or is it drifting? Where? In what direction?
- What structural rules does this codebase implicitly follow, and where are they being violated?
- What is the domain? What are the bounded contexts, the aggregates, the natural seams — not as declared in documentation, but as revealed by how the code is actually coupled?

The tool should feel like a structural X-ray of the codebase. The developer sees what was always there but invisible without the right lens.

**Structural intelligence as LLM context.** One of the most immediate uses: structural intelligence, formatted as a compact representation of the codebase's topology, becomes context for an LLM. An LLM that reads source code can assess local quality. An LLM that reads source code plus the structural X-ray can reason about global architecture. This means the output format of structural intelligence must be designed not only for human readability but for LLM ingestibility — compact, structured, rich enough to inform architectural reasoning, small enough to fit in a context window.

**End goals.** Structural intelligence is not the final product. It is the foundation for two higher-level outcomes:

First: structural intelligence combined with an LLM produces a concrete refactoring plan. The structural analysis identifies what's wrong (anomalies, violations, drift, overloaded bridges). The LLM translates this into specific, actionable refactoring steps — which functions to extract, which modules to split, which interfaces to introduce, in what order, with what dependencies between the steps. The structural analysis provides the what and where. The LLM provides the how and why.

Second: structural intelligence is used to extract the domain model — a graph schema in the DDD sense. Bounded contexts derived from structural modules. Aggregates derived from tightly coupled clusters. Relationships derived from cross-module edges. Entity types derived from structural roles. This schema is not a code artifact — it is a domain-level abstraction that transcends the codebase. It can be used to generate architecture diagrams, to onboard new developers, to compare the actual domain model against the intended one, and to guide future development by making the implicit domain structure explicit and enforceable.

---

### What We Don't Want

We don't want another dependency visualization tool. Dependency graphs exist. They show edges. They don't tell you what the edges mean structurally.

We don't want metrics dashboards. Cyclomatic complexity, line count, coupling scores — these are local measurements. They don't capture global structural position.

We don't want AI-generated code reviews. LLMs reading code can assess local quality. They cannot assess global structural health because they process text, not topology.

We don't want a tool that requires the developer to understand spectral graph theory, eigenvalues, or any mathematical concept. The math is internal. The output is structural insight in plain language.

---

### The Core Pipeline

The tool has a natural sequence of concerns, each building on the previous:

**Parsing.** Source code becomes a typed, multilayer graph. Nodes are code entities (functions, classes, modules). Edges are structural relationships (calls, imports, inheritance, co-location, temporal co-change). Each relationship type forms a separate layer of the graph. The parser must be precise — false edges corrupt everything downstream.

**Structural analysis.** The multilayer graph becomes structural fingerprints and derived insights. This is where spectral decomposition (or equivalent methods) extracts global topology. The output is not numbers for mathematicians — it is structural roles, module boundaries, anomalies, and constraints, expressed in terms developers understand.

**Interface.** Structural insights become developer-facing output. CLI reports, CI/CD checks, IDE integration. The tool fits into existing workflows rather than creating new ones.

Each concern is independent. The parser knows nothing about spectral analysis. The analysis knows nothing about the interface. This separation is load-bearing — it allows each concern to evolve independently and to be validated independently.

---

### The Key Bet

The bet is: spectral analysis of code graphs produces structurally meaningful results — that the mathematical decomposition aligns with architectural reality as understood by developers who know the codebase.

This is not guaranteed. Spectral methods work well on graphs with clear community structure (social networks, biological networks). Code graphs may or may not have this property. The directory structure that developers impose may or may not align with the topological structure that spectral analysis reveals. Spectral decomposition may or may not outperform simpler methods (community detection, directory grouping, degree-based heuristics).

The bet must be tested empirically before significant engineering investment. The cheapest possible test: take a well-documented open-source codebase, parse its call graph, run spectral clustering, compare against documented architecture. If the results align, the bet is validated. If not, the approach needs rethinking or the project stops.

Everything else in this project is conditional on this bet paying off.

---

### The Multilayer Dimension

A codebase is not one graph. It is several overlapping graphs on the same set of nodes. The call graph captures runtime coupling. The import graph captures compile-time coupling. The inheritance graph captures type coupling. The co-location graph captures organizational coupling. The co-change graph (from git history) captures evolutionary coupling.

A function can be central in the call graph but peripheral in the import graph. A class can be tightly coupled by inheritance but loosely coupled by actual usage. These cross-layer discrepancies are often the most structurally informative findings — they reveal where different kinds of coupling disagree, which is where architectural tension lives.

Whether spectral analysis should run independently per layer and combine results, or jointly on a fused multilayer representation, is an open question. The answer depends on whether cross-layer interactions produce meaningful spectral signatures that per-layer analysis misses. This should be tested empirically, not decided in advance.

---

### The Schema Question

The most ambitious goal: derive both a structural schema and a domain model from the codebase's own topology.

A structural schema is a set of topological constraints that the codebase implicitly follows. Not constraints imposed by a type system or linter — constraints that emerge from the actual graph structure. "Utility functions are leaves." "Each module has one primary entry point." "The call graph is approximately layered."

These constraints are not written down anywhere. They exist as emergent regularities in the topology. When they're violated, the violation often corresponds to a real architectural problem — tech debt, accidental coupling, design drift.

A domain model goes further: it maps the structural modules to domain concepts. If the structural analysis identifies a tightly coupled cluster of functions dealing with user credentials, session tokens, and permission checks, that cluster is the "authentication" bounded context — regardless of whether those functions are in a directory called "auth" or scattered across the codebase. The domain model is the structural schema elevated to semantic meaning, bridging the gap between topology and intent.

Whether spectral fingerprints are the right tool for either of these, or whether simpler graph statistics suffice, is an open question. The schema and domain model, if achievable, would function as a living architectural description — continuously verified, evolving with the codebase, and serving as the ground truth for what the system actually is versus what anyone thinks it is.

---

### Open Questions

These are the questions that should guide research and experimentation. They are ordered roughly by depth — foundational questions first, application questions later.

**What are the irreducible node kinds?** We assume functions, classes, and modules. But is this the right decomposition? A method is a function inside a class — is it a distinct kind or a function with a parent? A module in Python is a file, but in TypeScript it can be a namespace within a file. A class in some languages is just a namespace for functions. Perhaps the irreducible set is smaller: just "callable" and "namespace." Perhaps it's larger: "callable," "type," "namespace," "value," "interface." The right decomposition should emerge from what the structural analysis actually needs to distinguish, not from language grammar categories. Different kinds are justified only if they occupy structurally distinct positions in the graph — if collapsing two kinds into one loses structural information.

**What are the irreducible relationship layers?** We assume calls, imports, inheritance, co-location, and co-change. But these were chosen by intuition, not derived. Are some redundant? (Does co-location just approximate the import graph?) Are some missing? (What about data flow — "function A produces values consumed by function B"? What about interface implementation vs class inheritance?) The right set of layers is the minimal set where each layer carries structural information not captured by any combination of the others. This should be tested: add a layer, check if downstream structural analysis improves. If not, the layer is redundant.

**Does spectral analysis of code graphs produce architecturally meaningful clusters?** This is the foundational empirical question. If spectral fingerprint similarity doesn't correspond to architectural module membership on well-documented codebases, the approach needs fundamental rethinking.

**Does spectral analysis outperform simpler baselines?** Directory grouping is free and often good. Louvain community detection is cheap and well-understood. The spectral approach must demonstrably outperform these on architectural alignment to justify its complexity.

**Which graph layers contribute the most structural signal?** Is the call graph sufficient, or do imports, inheritance, co-location, and co-change add meaningful information? The answer determines how complex the parser needs to be.

**Do cross-layer discrepancies correspond to real architectural concerns?** If a function is central in one layer but peripheral in another, does this predict structural problems? This determines whether multilayer analysis is worth the effort.

**Can structural roles be classified from spectral fingerprints in a way that developers recognize as accurate?** "Hub," "bridge," "utility," "entry point" — do these categories emerge from the spectral data, and do developers agree with the classifications? Are these even the right categories, or does the data reveal a different taxonomy of structural roles?

**Are automatically detected anomalies useful?** If fewer than half the detected anomalies correspond to real structural concerns, the detection is too noisy to ship. The threshold for useful anomaly detection is high.

**Can structural constraints be extracted automatically, and do violations predict future problems?** This is the schema question. If violated constraints correlate with later bug fixes, refactoring PRs, or tech debt tickets, the schema is genuinely predictive. If not, it's descriptive at best.

**Does temporal analysis (structural drift across git history) reveal meaningful trends?** Architecture drift is a real concern in long-lived codebases. Whether spectral fingerprint comparison across time snapshots captures this usefully is an empirical question.

**Can the structural modules be mapped to domain concepts?** This is the DDD question. If the structurally derived modules correspond to recognizable domain concepts (authentication, billing, inventory), then the tool can extract domain models. If the structural modules don't align with domain boundaries, structural intelligence and domain modeling may need different approaches.

**What representation of structural intelligence is most useful as LLM context?** A full spectral fingerprint matrix is meaningless to an LLM. A natural language summary loses precision. The right representation is somewhere between — structured enough to be precise, semantic enough to be interpretable. Finding this format is a design problem, not a math problem.

---

### Methodology

Every claim the tool makes must be empirically validated against developer judgment on real codebases. The tool's structural findings are hypotheses about architecture. Developers who know the codebase are the ground truth.

The validation methodology:

- Use well-documented open-source codebases where architectural intent is known.
- Compare tool output against documented architecture, known refactoring decisions, and developer expert judgment.
- Always compare against the simplest baseline that could work (directory grouping, degree counting, Louvain clustering).
- A finding that doesn't outperform the simple baseline is not a finding.
- Developer agreement is the ultimate metric. If developers who know the code say "this is wrong" or "this is obvious," the tool has failed on that finding, regardless of what the math says.

The tool improves through a cycle: run analysis → present to developers → collect feedback on accuracy and usefulness → adjust the analysis pipeline → repeat. The methodology is empirical and iterative, not theoretically derived.

---

### What Remains Permanently True

Regardless of implementation choices, these hold:

- A codebase has both a textual and a topological description. Current tools exploit the textual description far more than the topological one.
- Global structural properties of the topology (module boundaries, structural roles, bottlenecks, drift) are invisible to local analysis and valuable to developers.
- Spectral decomposition is the established mathematical method for extracting global structure from graphs. Whether it's the right method for code graphs specifically is an empirical question, not a theoretical one.
- The value of the tool is in the structural insight it surfaces, not in the mathematical method it uses internally. If a simpler method produces equally good insight, use the simpler method.
- Developer judgment is the ground truth. The math serves the developer, not the other way around.

---
