# Diagnostics

This is the authoritative reference for every diagnostic topo produces. There are exactly 13 diagnostics across three phases. No other diagnostics will be added without updating this document.

**Phases:**
- **Phase 1** (3 diagnostics): Mathematical graph facts. Computed from the dependency graph alone. No inference, no semantics.
- **Phase 1+** (4 diagnostics): Structural inference. Still graph-only, but requires derived quantities (Fiedler vectors, inferred layering, git history).
- **Phase 2** (4 diagnostics): Structural-semantic disagreement. Requires code embeddings from jina-embeddings-v2-base-code via fastembed-rs. Behind the `semantic` Cargo feature flag.
- **Phase 3** (2 diagnostics): Learned structural intelligence. Requires trained R-GIN model. Replaces or upgrades Phase 2 diagnostics where uplift is confirmed.

**Output shape:** Every diagnostic produces an `IssueOutput` with id, kind, title, description, severity (0.0-1.0), severity_label, confidence (0.0-1.0), confidence_label, and anchors (node locations).

---

## Table of Contents

| # | ID | Phase | Absorbs |
|---|---|---|---|
| 1 | `cycle-member` | 1 | — |
| 2 | `wide-interface` | 1 | — |
| 3 | `cross-package-coupling` | 1 | — |
| 4 | `near-disconnect` | 1+ | — |
| 5 | `overloaded-utility` | 1+ | `fragile-hub` |
| 6 | `layer-violation` | 1+ | `reverse-dependency`, `bidirectional-dependency` |
| 7 | `patch-magnet` | 1+ | — |
| 8 | `misplaced-concern` | 2 | — |
| 9 | `incoherent-module` | 2 | `god-module`, `low-cohesion` |
| 10 | `shadow-dependency` | 2 | — |
| 11 | `scattered-api` | 2 | — |
| 12 | `misplaced-concern` (upgraded) | 3 | Phase 2 `misplaced-concern` |
| 13 | `coupling-mismatch` | 3 | `layer-discrepancy` (never implemented) |

---

## Diagnostic 1: `cycle-member`

**Phase:** 1 | **Absorbs:** — | **Replaced by:** —

### First-Principles Justification

Software should form a directed acyclic graph at the module level. Cycles form when two modules evolve together under time pressure and each takes a shortcut through the other. Once a cycle exists, neither module can be compiled, tested, or reasoned about independently. The engineering force is **mutual convenience under deadline**: A calls B because B already has the data, then B calls A because A already has the logic.

### Root Causes

**1. Layering violation hardened into a cycle.**
Module A depends on B (correct direction). Under pressure, B adds a call back to A. Over time both directions accumulate edges.
- *Confirm:* One direction has significantly fewer edges than the other (asymmetric cycle).
- *Action:* Invert the minority-direction edges by extracting a shared interface or moving the called function down.

**2. Shared mutable state.**
Two modules both read and write the same data structure, creating implicit bidirectional coupling.
- *Confirm:* Cycle edges pass through a shared struct/class that both modules import and mutate.
- *Action:* Extract the shared state into a third module that both depend on unidirectionally.

**3. Collocated concerns that should be one module.**
The "two modules" are actually one domain concept split across two packages for historical reasons.
- *Confirm:* Nodes in both modules are semantically coherent (same domain vocabulary). Module boundary aligns with a directory split, not a domain split.
- *Action:* Merge the modules.

### Detection

```
Input: directed graph G (all edge types combined)
Compute Tarjan's SCCs on G
For each SCC with |nodes| >= 2:
  participating_modules = distinct modules of SCC members
  For each module pair (A, B) where both have members in this SCC:
    edges_A_to_B = count of directed edges from A-members to B-members
    edges_B_to_A = count of directed edges from B-members to A-members
    Emit cycle-member(scc_nodes, module_pair, edges_A_to_B, edges_B_to_A)
```

### Severity Model

```
severity = 0.4 * size_factor + 0.3 * module_span_factor + 0.3 * depth_factor

size_factor        = clamp(|scc_nodes| / 20, 0, 1)
module_span_factor = clamp(|participating_modules| / 5, 0, 1)
depth_factor       = clamp(longest_path_in_scc / 10, 0, 1)

Multipliers:
  SCC contains entry points: x1.3
  SCC spans >2 declared packages: x1.4
  SCC size > p90 of all SCCs: x1.2
```

### Example Output

```
cycle-member
  Symptom: 8 nodes form a dependency cycle spanning auth and session modules.
  Impact: Neither module can be compiled, tested, or refactored independently.
    Changes to either module risk cascading through the cycle.
  Investigate: The cycle has 5 edges auth->session and 2 edges session->auth.
    Check whether the 2 session->auth edges can be inverted by extracting
    a shared interface.
  Evidence:
    SCC size: 8 nodes
    Modules: auth (5 nodes), session (3 nodes)
    Forward edges (auth->session): 5
    Backward edges (session->auth): 2
    Anchors: session::validate -> auth::check_token, session::refresh -> auth::renew
```

### False Positive Suppression

- **Trait/interface cycles:** If all cycle edges pass through a trait impl or interface implementation (detected by edge type = `inherits`), suppress. Type hierarchies commonly form small cycles through mutual trait bounds.
- **Test-only cycles:** If >80% of cycle members have paths containing `test`/`spec`/`_test`, suppress.
- **Trivial 2-node cycles:** If SCC has exactly 2 nodes and both are in the same declared package, demote severity by 0.5x (likely collocated concerns, low blast radius).

### Interactions

- **`layer-violation`:** If both modules in a cycle also trigger layer-violation, suppress layer-violation on that pair. The cycle is the larger problem; the layer violation is a symptom.
- **`cross-package-coupling`:** Cycles that span packages are compound problems. Both fire independently.

---

## Diagnostic 2: `wide-interface`

**Phase:** 1 | **Absorbs:** — | **Replaced by:** —

### First-Principles Justification

When two modules communicate through many distinct coupling points (functions, types, constants), changes in one module's internals propagate to the other through a wide surface area. The engineering force is **incremental coupling**: each new feature adds one more cross-module call, and nobody notices the interface growing because each addition is small. Wide interfaces resist refactoring because every coupling point is a potential break.

### Root Causes

**1. Missing facade.**
Module A directly calls 15 internal functions of module B instead of going through a narrow public API.
- *Confirm:* The coupling points in B are private/internal functions (not marked `pub` in Rust, not in `__init__.py` in Python, not exported in JS).
- *Action:* Introduce a facade in B that exposes the 3-4 operations A actually needs.

**2. Feature creep across a boundary.**
Two modules started with a clean interface. Successive features each added one more cross-module call.
- *Confirm:* Git history shows coupling point count growing monotonically over time.
- *Action:* Audit which coupling points are still used. Remove stale ones. Group the rest behind a narrower interface.

**3. Modules that should be merged.**
The two modules are so tightly coupled that the boundary is artificial.
- *Confirm:* Coupling is approximately symmetric (both modules call each other through many points). Semantic content overlaps.
- *Action:* Merge the modules or reorganize along a different seam.

### Detection

```
Input: graph G, module assignments
For each ordered module pair (A, B):
  coupling_points = distinct (source_node, target_node) edges from A to B
  |coupling_points| = width
Compute Tukey fence: Q1, Q3, IQR over all pair widths where width > 0
threshold = Q3 + 1.5 * IQR
For each pair where width > threshold AND width >= 5:
  Emit wide-interface(A, B, width, threshold, top_coupling_points)
```

### Severity Model

```
severity = 0.5 * excess_factor + 0.3 * asymmetry_factor + 0.2 * concentration_factor

excess_factor        = clamp((width - threshold) / threshold, 0, 1)
asymmetry_factor     = 1.0 - min(width_A_to_B, width_B_to_A) / max(width_A_to_B, width_B_to_A)
                       (1.0 if unidirectional, 0.0 if perfectly symmetric)
concentration_factor = max_target_in_degree / width
                       (how much coupling concentrates on a single target node)

Multipliers:
  Both directions exceed threshold: x1.3 (bidirectional wide interface)
  Module pair is in same declared package: x0.7 (internal coupling is less alarming)
```

### Example Output

```
wide-interface
  Symptom: 23 distinct coupling points from billing to auth (threshold: 11).
  Impact: Changes to auth internals have 23 potential break points in billing.
    This interface is wider than any other module pair in the codebase.
  Investigate: Are all 23 coupling points necessary? Check whether billing
    could use a facade in auth instead of calling internal functions directly.
  Evidence:
    Width: 23 (threshold: 11, Q3: 8, IQR: 2)
    Top targets in auth: validate_token (called 6x), get_permissions (called 4x),
      check_role (called 3x)
    Reverse width (auth->billing): 2
```

### False Positive Suppression

- **Re-export modules:** If module B is a barrel/re-export module (>80% of its nodes are re-exports of other modules), suppress. The wide interface is an artifact of the re-export pattern.
- **Standard library facades:** If module B matches patterns like `utils`, `helpers`, `common`, `shared`, and has in-degree from >60% of all modules, suppress (it is a designed utility module).
- **Minimum module size:** Suppress if either module has <5 nodes (small modules inflate width relative to size).

### Interactions

- **`overloaded-utility`:** If the top target node in the wide interface also triggers overloaded-utility, the combination indicates a single function absorbing too much cross-module traffic. Both fire independently; the developer should address the utility first.
- **`incoherent-module`:** A wide interface into an incoherent module suggests the module should be split, which would naturally narrow the interface.

---

## Diagnostic 3: `cross-package-coupling`

**Phase:** 1 | **Absorbs:** — | **Replaced by:** —

### First-Principles Justification

Spectral clustering discovers the actual structural modules of a codebase from the topology of its dependency graph. Declared packages represent the developer's intended module boundaries. When a single spectral module spans multiple declared packages, the code is more tightly coupled across package boundaries than within them. The engineering force is **organizational inertia**: the package structure reflects a past design decision, but the code has evolved to couple across that boundary.

### Root Causes

**1. Package split that doesn't match actual coupling.**
A refactoring split one package into two, but the code in both packages still forms a single tightly-coupled cluster.
- *Confirm:* The spectral module's internal edge density is much higher than expected for two independent packages. Git history shows the packages were recently split.
- *Action:* Either complete the decoupling (reduce cross-package edges) or undo the split.

**2. Shared domain concept split across packages.**
Authentication logic lives in both `auth` and `user` packages because the domain concept spans both.
- *Confirm:* Nodes in both packages share domain vocabulary (same naming patterns, same types referenced).
- *Action:* Extract the shared concern into its own package, or consolidate into one package.

**3. Spectral clustering artifact.**
The codebase has weak overall modularity and the spectral clustering is noisy.
- *Confirm:* Silhouette score is low (<0.3). Multiple spectral modules span multiple packages.
- *Action:* Not actionable per se. Consider that the codebase may not have strong module boundaries at all.

### Detection

```
Input: spectral module assignments, declared package assignments
For each spectral module M:
  packages_in_M = distinct declared packages of M's members
  If |packages_in_M| >= 2:
    For each package P in packages_in_M:
      fraction_of_P_in_M = |nodes in M ∩ P| / |nodes in P|
    dominant_package = argmax fraction_of_P_in_M
    cross_package_nodes = nodes in M not in dominant_package
    Emit cross-package-coupling(M, packages_in_M, cross_package_nodes, fractions)
```

### Severity Model

```
severity = 0.4 * span_factor + 0.3 * balance_factor + 0.3 * density_factor

span_factor    = clamp((|packages_in_M| - 1) / 4, 0, 1)
balance_factor = 1.0 - (|dominant_package_nodes| / |M|)
                 (0 if one package dominates, 1 if evenly split)
density_factor = cross_package_edge_density / total_module_edge_density
                 (how much of the module's coupling is cross-package)

Multipliers:
  Silhouette score > 0.5 for this module: x1.3 (clustering is confident)
  Module contains entry points from multiple packages: x1.2
  Package fallback was NOT used (spectral clustering succeeded): x1.2
```

### Example Output

```
cross-package-coupling
  Symptom: Spectral module 3 spans packages auth and user.
    8 of 12 nodes are in auth, 4 are in user.
  Impact: The declared package boundary between auth and user does not match
    the actual structural coupling. Code in these packages evolves as a unit
    despite being organizationally separate.
  Investigate: Are the 4 user nodes (user::validate_credentials,
    user::session_from_token, user::permission_check, user::role_lookup)
    conceptually part of auth? If so, move them. If not, decouple them
    from the auth cluster.
  Evidence:
    Spectral module 3: 12 nodes, silhouette: 0.61
    Packages: auth (8 nodes, 67%), user (4 nodes, 33%)
    Cross-package edges: 9 of 14 module-internal edges (64%)
```

### False Positive Suppression

- **Package fallback active:** If spectral clustering was not used (silhouette too low, fell back to packages), suppress all cross-package-coupling diagnostics. The modules ARE the packages; the diagnostic is tautologically impossible.
- **Single-node spillover:** If only 1 node from a secondary package is in the module, suppress (likely a bridge node, not a coupling problem).
- **Low silhouette module:** If the module's silhouette score < 0.2, suppress (clustering is uncertain for this module).

### Interactions

- **`cycle-member`:** Cross-package coupling often co-occurs with cycles between the same packages. Both fire independently; the cycle is the more urgent problem.
- **`misplaced-concern` (Phase 2):** The cross-package nodes are candidates for misplaced-concern once semantic analysis is available. Cross-package-coupling identifies the structural signal; misplaced-concern adds semantic confirmation.

---

## Diagnostic 4: `near-disconnect`

**Phase:** 1+ | **Absorbs:** — | **Replaced by:** —

### First-Principles Justification

Codebases accrete features without architectural review. Over time, some region communicates with the rest through one or two functions. The Fiedler value (algebraic connectivity, lambda_2 of the graph Laplacian) measures this: when it approaches zero, the graph is one deletion away from splitting into disconnected components. The engineering force is **accretion without review** -- modules drift apart and nobody notices the coupling has narrowed to a thread.

### Root Causes

**1. Intentional architectural boundary.**
Two subsystems were designed for isolation. A facade connects them through narrow entry points.
- *Confirm:* Bridge nodes are explicit interface types (traits, abstract classes). High betweenness, low in-degree.
- *Action:* Not a problem. Annotate as architectural boundary. Formalize as a package boundary if not already.

**2. Organic drift into near-disconnection.**
Two regions were once tightly coupled, but incremental refactoring removed most cross-cutting edges.
- *Confirm:* Git history shows cross-module edges were more numerous in earlier commits. Bridge nodes are adapters with low semantic coherence to either side.
- *Action:* Either complete the separation (extract into separate packages) or re-establish coupling through a proper interface.

**3. Dead code limb.**
A subsystem is effectively unused but technically reachable through one call path.
- *Confirm:* Bridge node has exactly one caller. The subtree on the far side has no entry points or tests.
- *Action:* Remove the dead subtree or explicitly deprecate it.

### Detection

```
Input: graph Laplacian L of largest connected component
Compute eigenvalues; lambda_2 = Fiedler value, v_2 = Fiedler vector

Per-module fragility (primary signal):
  For each module M with |M| >= 5:
    L_M = subgraph Laplacian restricted to M's nodes
    lambda_2_M = Fiedler value of L_M
    If lambda_2_M < module_fiedler_threshold (p10 of all module lambda_2 values):
      Partition M's nodes by sign of v_2_M
      cut_edges_M = edges crossing the partition within M
      bridge_nodes_M = nodes incident to cut_edges_M, sorted by betweenness desc
      Emit near-disconnect(M, partitions, bridge_nodes_M, lambda_2_M)

Global fragility (secondary signal):
  If lambda_2 < global_fiedler_threshold:
    Partition all nodes by sign of v_2
    cut_edges = edges crossing the partition
    bridge_nodes = nodes incident to cut_edges, sorted by betweenness desc
    Emit near-disconnect(global, partitions, bridge_nodes, lambda_2)
```

### Severity Model

```
severity = 0.4 * fiedler_factor + 0.3 * cut_ratio_factor + 0.3 * size_balance_factor

fiedler_factor      = 1.0 - clamp(lambda_2 / fiedler_median, 0, 1)
cut_ratio_factor    = 1.0 - clamp(|cut_edges| / expected_cut, 0, 1)
                      where expected_cut = (2 * |E| * s1 * s2) / (n * (n - 1))
size_balance_factor = min(s1, s2) / max(s1, s2)

Multipliers:
  Bridge node betweenness > p90: x1.3 (single point of failure)
  Smaller partition contains entry points: x1.2
  Both partitions in same declared package: x1.5
```

### Example Output

```
near-disconnect
  Symptom: Module auth is structurally fragile between token_validation
    (5 nodes) and credential_storage (7 nodes). Only 1 edge connects them.
  Impact: Removing or modifying auth::token_bridge would structurally
    disconnect 5 nodes from the rest of the module.
  Investigate: Is this an intentional internal boundary? If so, consider
    splitting into two modules. If not, the coupling has eroded — restore
    it or complete the separation.
  Evidence:
    Fiedler value (module): 0.002 (module median: 0.14)
    Cut edges: 1 (expected for random partition: 6)
    Bridge node: auth::token_bridge (betweenness: 0.41)
    Partition A: token_validation (5 nodes)
    Partition B: credential_storage (7 nodes)
```

### False Positive Suppression

- **Plugin/extension architectures:** If the bridge node matches `plugin`/`extension`/`register` patterns, or if >3 subgraphs all connect through the same hub (star topology), suppress.
- **Test subgraphs:** If >80% of nodes on the smaller side have paths containing `test`/`spec`/`_test`, suppress.
- **Small graphs:** Suppress if the partition being analyzed has <10 nodes (too small for meaningful Fiedler analysis).

### Interactions

- **`overloaded-utility`:** If the bridge node is also an overloaded-utility, the combination is critical -- a single overloaded function is the only structural link. Both fire independently.
- **`layer-violation`:** A near-disconnect that aligns with a layer boundary is more likely intentional. If bridge nodes are at a layer boundary, reduce severity by 0.7x.

---

## Diagnostic 5: `overloaded-utility`

**Phase:** 1+ | **Absorbs:** `fragile-hub` | **Replaced by:** —

### First-Principles Justification

Utility functions accumulate responsibilities through the gravity of convenience -- adding one more line to a widely-called function is cheaper than creating a new one. Each addition is locally rational. The cumulative result is a directional bottleneck: high in-degree (many callers), low out-degree (few callees), callers spanning many modules. This is not a hub (balanced in/out connecting peers); it is a sink that absorbs cross-cutting concerns. The engineering force is **reuse without abstraction**.

### Root Causes

**1. God utility.**
A function grew beyond its original scope by accumulating unrelated responsibilities.
- *Confirm:* Callers span multiple semantic domains. Function body mixes concerns (if semantic embeddings available: low self-coherence).
- *Action:* Split along responsibility boundaries.

**2. Missing abstraction layer.**
The utility is the only common ground between subsystems that should share an interface.
- *Confirm:* Callers cluster into distinct groups that each use different subsets of the utility's functionality.
- *Action:* Introduce a shared trait/interface. Each caller group gets a dedicated implementation.

**3. Legitimate standard library function.**
The function is intentionally called by everything (e.g., `log`, `format`, `serialize`).
- *Confirm:* Function name matches standard patterns. Signature is stable and generic.
- *Action:* Not a problem. Suppress.

### Detection

```
Input: directed graph G, module assignments
For each node v:
  direction_score = (in_degree(v) - out_degree(v)) / (in_degree(v) + out_degree(v))
  caller_modules = |distinct modules of nodes with edge -> v|
  caller_diversity = caller_modules / total_modules

If in_degree(v) > p85
   AND direction_score > 0.5
   AND caller_diversity > 0.3:
  Emit overloaded-utility(v, in_degree, out_degree, caller_modules, direction_score)
```

### Severity Model

```
severity = 0.4 * degree_factor + 0.3 * diversity_factor + 0.3 * direction_factor

degree_factor    = clamp((in_degree - p85_threshold) / (p99_threshold - p85_threshold), 0, 1)
diversity_factor = clamp(caller_diversity / 0.7, 0, 1)
direction_factor = clamp((direction_score - 0.5) / 0.5, 0, 1)

Multipliers:
  Node also has betweenness > p90: x1.3 (structural bridge + bottleneck)
  Node is in a utility/shared/common module: x0.7 (expected position)
  Node has out_degree == 0 (pure leaf): x0.8 (less likely to propagate changes)
```

### Example Output

```
overloaded-utility
  Symptom: utils::format_response is called by 34 nodes across 7 of 9 modules,
    but only calls 2 functions itself.
  Impact: Any change to format_response propagates to 34 callers across 7 modules.
    It is a directional bottleneck whose modification has outsized blast radius.
  Investigate: Does this function have a single responsibility, or has it
    accumulated unrelated formatting concerns? Check whether callers use
    different subsets of its behavior.
  Evidence:
    In-degree: 34 (p92), out-degree: 2
    Direction score: 0.89
    Caller modules: 7 of 9 (78%)
    Top callers: billing::render_invoice (3 calls), auth::format_error (2 calls)
```

### False Positive Suppression

- **Logging/tracing functions:** If node name matches `log`/`trace`/`debug`/`info`/`warn`/`error`/`println`/`print`, suppress.
- **Serialization/deserialization:** If node name matches `serialize`/`deserialize`/`to_json`/`from_json`/`encode`/`decode`, suppress.
- **Constructor/factory patterns:** If node name matches `new`/`create`/`build`/`from`/`default`, suppress.
- **Public API surface:** If the node is annotated as a public API entry point (detected by export analysis), suppress.

### Interactions

- **`near-disconnect`:** If the overloaded-utility is also a bridge node in a near-disconnect, the combination is critical. Both fire; the developer should address the bridge-bottleneck first.
- **`wide-interface`:** The overloaded-utility may be the dominant target in a wide interface. If so, narrowing the utility's responsibilities will also narrow the interface.
- **`patch-magnet`:** Overloaded utilities that are also patch magnets have the worst risk profile: high blast radius AND high change frequency.

---

## Diagnostic 6: `layer-violation`

**Phase:** 1+ | **Absorbs:** `reverse-dependency`, `bidirectional-dependency` | **Replaced by:** —

### First-Principles Justification

Well-structured codebases have a directional flow: higher-level modules depend on lower-level modules, not the reverse. This layering is rarely documented but is implicitly maintained by convention. Violations occur when a lower-level module reaches up to call a higher-level one, creating a dependency that inverts the intended direction. The engineering force is **expedience**: calling the function that already exists is faster than creating a proper callback, event, or interface. Each violation is small, but accumulated violations destroy the codebase's ability to reason about dependency direction.

### Root Causes

**1. Callback avoidance.**
A low-level module needs to notify a high-level one. Instead of accepting a callback or emitting an event, it directly imports and calls the high-level function.
- *Confirm:* The upward edge is a single call from an otherwise-leaf module to a high-level orchestrator. The low-level module has no other upward edges.
- *Action:* Replace with a callback parameter, event, or observer pattern.

**2. Circular feature dependency.**
Two features at different layers depend on each other's data. Neither team wants to own the shared abstraction.
- *Confirm:* Multiple upward edges between the same pair. Both directions carry significant traffic. Often accompanied by `cycle-member` on the same pair.
- *Action:* Extract the shared data into a lower layer that both depend on.

**3. Misclassified module layer.**
The inferred layering is wrong. The "lower" module is actually a peer, not a dependency.
- *Confirm:* The module has roughly equal in-degree and out-degree to the "higher" module. The inferred layer assignment is based on a narrow edge majority.
- *Action:* Not a true violation. Adjust the layer inference or suppress.

### Detection

```
Input: directed graph G, module assignments
Step 1 — Infer layer ordering:
  For each module pair (A, B):
    edges_A_to_B = count of A -> B edges
    edges_B_to_A = count of B -> A edges
  Construct a module-level DAG by keeping the majority direction for each pair.
  Assign layers via topological sort of the module DAG.
  (Ties broken by: module with higher total out-degree is placed higher.)

Step 2 — Flag violations:
  For each edge (u, v) where layer(module(u)) < layer(module(v)):
    // u is in a lower layer calling v in a higher layer = upward dependency
    minority_ratio = edges_low_to_high / (edges_low_to_high + edges_high_to_low)
    If minority_ratio < 0.4:
      // This direction is genuinely the minority
      Emit layer-violation(u, v, layer(module(u)), layer(module(v)), minority_ratio)

Step 3 — Role-based violations:
  For each edge (u, v):
    If role(u) == "entry_point" AND role(v) == "utility" AND direction is upward:
      Emit layer-violation(u, v, "structural impossibility: entry calling utility upward")
```

### Severity Model

```
severity = 0.4 * minority_factor + 0.3 * layer_gap_factor + 0.3 * edge_count_factor

minority_factor   = 1.0 - minority_ratio
                    (stronger signal when the violating direction is a smaller fraction)
layer_gap_factor  = clamp((layer_high - layer_low) / max_layer_depth, 0, 1)
                    (violations spanning more layers are worse)
edge_count_factor = clamp(|violating_edges_this_pair| / 5, 0, 1)

Multipliers:
  Both modules in same SCC: suppress entirely (cycle-member covers it)
  Violating edge is the only edge between the modules: x1.3 (fragile + wrong direction)
  Role-based impossibility: override severity to 0.8 minimum
```

### Example Output

```
layer-violation
  Symptom: 3 edges from database (layer 1) call up to api_handlers (layer 3).
    The dominant direction is api_handlers -> database (41 edges).
  Impact: The database layer has upward knowledge of HTTP handler internals.
    This prevents the database layer from being used independently.
  Investigate: Check database::notify_handler, database::format_for_api,
    database::log_request. Are these convenience shortcuts that should be
    callbacks or events instead?
  Evidence:
    Upward edges: 3 (database -> api_handlers)
    Downward edges: 41 (api_handlers -> database)
    Minority ratio: 0.068
    Layer gap: 2 (layer 1 -> layer 3)
```

### False Positive Suppression

- **Balanced pairs:** If minority_ratio >= 0.4, suppress (the pair is roughly symmetric; there is no clear layering).
- **Same SCC:** If both modules are in the same strongly connected component, suppress layer-violation. The cycle-member diagnostic covers this.
- **Event/callback edges:** If the violating edge target matches `on_*`/`handle_*`/`callback`/`listener`/`observer` naming patterns, suppress (the "upward" call is an intentional inversion-of-control pattern).
- **Type-only edges:** If the violating edge is an import of a type (struct/enum/trait) with no function call, demote severity by 0.5x (type imports across layers are common and low-risk).

### Interactions

- **`cycle-member`:** If both modules in the violation are in the same SCC, suppress this diagnostic. The cycle is the bigger problem.
- **`wide-interface`:** Layer violations through a wide interface compound the problem: not only is the direction wrong, but the coupling surface is large. Both fire independently.
- **`cross-package-coupling`:** Layer violations across package boundaries are more severe than within a package.

---

## Diagnostic 7: `patch-magnet`

**Phase:** 1+ | **Absorbs:** — | **Replaced by:** — | **Flag:** `--git`

### First-Principles Justification

Some nodes are structurally peripheral (low centrality, few dependents) yet accumulate a disproportionate share of git commits. This mismatch indicates code that is unstable despite not being structurally important -- it changes often because it is poorly designed, under-specified, or absorbing requirements churn. The engineering force is **requirement instability concentrated in implementation details**: the node keeps changing because it encodes a volatile business rule or an unstable external interface.

### Root Causes

**1. Volatile business rule.**
The node encodes a rule that changes with every product iteration (pricing logic, feature flags, validation rules).
- *Confirm:* Git log shows changes correlating with product releases. Commit messages reference tickets/stories.
- *Action:* Extract the volatile logic into a configuration or rule engine. The structural code should be stable; the rules should be data.

**2. Bug-prone implementation.**
The node has latent bugs that surface periodically, causing repeated fix commits.
- *Confirm:* Git log shows fix/patch/bugfix commit messages. Changes are small and localized.
- *Action:* Rewrite with better test coverage. The high churn signals low code quality in this specific node.

**3. External interface instability.**
The node wraps an external API or dependency that changes its contract frequently.
- *Confirm:* Changes coincide with dependency upgrades. The node imports external crates/packages.
- *Action:* Introduce an anti-corruption layer. Isolate the external dependency behind a stable internal interface.

### Detection

```
Input: graph G, module assignments, git log (requires --git flag)

For each node v:
  commit_count = number of distinct commits touching v's file+range
  change_velocity = commit_count / repo_age_months
  structural_centrality = betweenness_percentile(v)

  churn_ratio = change_velocity / median_change_velocity
  centrality_ratio = structural_centrality

If churn_ratio > 3.0 AND centrality_ratio < 0.3:
  // High churn, low structural importance
  Emit patch-magnet(v, commit_count, change_velocity, structural_centrality)
```

### Severity Model

```
severity = 0.5 * churn_factor + 0.3 * peripherality_factor + 0.2 * recency_factor

churn_factor        = clamp((churn_ratio - 3.0) / 7.0, 0, 1)
peripherality_factor = 1.0 - structural_centrality
recency_factor      = fraction of commits in last 3 months vs total commits
                      (recent churn is more concerning than historical)

Multipliers:
  Node is in a utility/shared module: x0.7 (utilities are expected to change)
  Node has >3 distinct authors: x1.2 (multiple people struggling with it)
  Node is in a cycle (cycle-member): x1.3 (churn + cycle = compounding instability)
```

### Example Output

```
patch-magnet
  Symptom: billing::calculate_tax has been modified in 47 commits (5.2x median)
    but is structurally peripheral (betweenness: p12).
  Impact: Disproportionate engineering time is spent on a low-centrality node.
    Each change risks introducing bugs, but the node's peripheral position means
    these bugs surface late.
  Investigate: Is this encoding a volatile business rule that should be
    externalized as configuration? Or is the implementation fragile and
    in need of a rewrite?
  Evidence:
    Commits: 47 (median: 9, ratio: 5.2x)
    Betweenness percentile: p12
    Recent commits (last 3 months): 12 of 47 (26%)
    Authors: 4 distinct
```

### False Positive Suppression

- **Generated files:** If the node's file matches `*_generated.*`, `*.pb.*`, `*_pb2.py`, `*.g.dart`, suppress.
- **Configuration files:** If the node's file matches `config.*`, `settings.*`, `*.toml`, `*.yaml`, suppress.
- **Test files:** If the node's path contains `test`/`spec`/`_test`, suppress.
- **Very recent files:** If the file was created within the last 30 days, suppress (new files naturally have high change velocity).

### Interactions

- **`overloaded-utility`:** A patch-magnet that is also an overloaded-utility is the highest-risk combination: high blast radius AND high change frequency. Both fire independently; the developer should stabilize the utility first.
- **`misplaced-concern` (Phase 2):** A patch-magnet that is semantically misplaced may be churning because it is in the wrong module and keeps getting patched to bridge the gap.

---

## Diagnostic 8: `misplaced-concern`

**Phase:** 2 | **Absorbs:** — | **Replaced by:** Phase 3 `misplaced-concern` (upgraded)

### First-Principles Justification

A function can end up in the wrong module through copy-paste, expedient placement during a deadline, or organic code migration. Structurally, it looks like it belongs where it is (it is called by its neighbors, imported by its module). Semantically, it does something unrelated to its module. The engineering force is **placement by proximity instead of by purpose**: the function was created near the code that first needed it, not near the code it conceptually belongs with.

### Root Causes

**1. Expedient placement.**
A developer needed auth logic while working in the billing module and wrote it there.
- *Confirm:* Node's semantic embedding is closer to a different module's centroid. Node's name/content uses vocabulary from the other domain.
- *Action:* Move the node to the module it is semantically closest to.

**2. Module boundary drift.**
The function was correctly placed when the module had a different scope, but the module evolved and the function didn't move.
- *Confirm:* The module's semantic centroid has shifted away from this node over time. Other nodes in the module are semantically coherent; this one is the outlier.
- *Action:* Move the node to its semantic home.

**3. Cross-cutting concern without a home.**
The function is legitimately needed by its current module but its purpose is cross-cutting (logging, serialization, error formatting).
- *Confirm:* Node is semantically equidistant from multiple modules. No single module is a clear better home.
- *Action:* Extract into a shared/utility module. Or leave in place if the structural coupling is correct.

### Detection

```
Input: semantic embeddings S (768d), spectral module assignments, module centroids

For each node v in module M:
  sim_own = cosine_similarity(S[v], centroid(S[M]))
  For each other module M':
    sim_other = cosine_similarity(S[v], centroid(S[M']))
  best_other = argmax sim_other
  sim_best = max(sim_other)
  gap = sim_best - sim_own

If gap > 0.15 AND sim_own < 0.4:
  Emit misplaced-concern(v, M, best_other, sim_own, sim_best, gap)
```

### Severity Model

```
severity = 0.5 * gap_factor + 0.3 * confidence_factor + 0.2 * isolation_factor

gap_factor        = clamp(gap / 0.4, 0, 1)
confidence_factor = clamp(sim_best / 0.8, 0, 1)
                    (how strongly the node belongs to the other module)
isolation_factor  = 1.0 - (|structural_edges_to_own_module| / |total_edges|)
                    (structurally isolated from own module = more likely misplaced)

Multipliers:
  Node has >3 structural edges to best_other module: x1.3 (already coupled to destination)
  Node role is "bridge": x0.7 (bridges are expected to be between modules)
  Node is only misplaced concern in its module: x1.0 (individual case)
  >40% of module triggers misplaced-concern: suppress individuals, fire incoherent-module instead
```

### Example Output

```
misplaced-concern
  Symptom: billing::validate_jwt_token is semantically closer to module auth
    than to its own module billing.
  Impact: This function's purpose (JWT validation) is unrelated to billing.
    Developers looking for auth logic won't find it here. Changes to auth
    conventions won't naturally propagate to this function.
  Investigate: Does this function belong in auth? Check whether moving it
    would break billing's internal coupling or whether billing only calls
    it through a single entry point.
  Evidence:
    Similarity to own module (billing): 0.23
    Similarity to best module (auth): 0.71
    Gap: 0.48
    Structural edges to billing: 2, to auth: 4
```

### False Positive Suppression

- **Bridge nodes:** If the node's role is "bridge" and it has significant edges to both its own module and the suggested module, suppress (bridges are supposed to sit between modules).
- **Utility modules:** If the node's current module is a utility/shared/common module, suppress (utility modules are expected to contain semantically diverse functions).
- **Low embedding confidence:** If the node's source text is <50 tokens (too short for meaningful embedding), suppress.
- **Near-equal alternatives:** If the top 2 alternative modules have sim_other within 0.05 of each other, suppress (no clear single destination).

### Interactions

- **`incoherent-module`:** If >40% of a module's members trigger misplaced-concern, suppress the individual misplaced-concern diagnostics and fire incoherent-module instead. The module itself is the problem, not individual nodes.
- **`cross-package-coupling`:** A misplaced concern whose suggested module is in a different package is higher confidence -- the structural signal (cross-package coupling) and the semantic signal (misplaced concern) agree.
- **Phase 3 upgrade:** When Phase 3 `misplaced-concern` is active, suppress all Phase 2 `misplaced-concern` diagnostics. Phase 3 uses R-GIN reconstruction error instead of centroid distance, which is multi-hop, cross-layer, role-aware, and cross-repo calibrated.

---

## Diagnostic 9: `incoherent-module`

**Phase:** 2 | **Absorbs:** `god-module`, `low-cohesion` | **Replaced by:** —

### First-Principles Justification

A module can be structurally coupled (all its members call each other) but semantically incoherent (its members do unrelated things). This happens when coupling is accidental -- shared state, convenience imports, or a common dependency forces unrelated functions into the same cluster. The spectral clustering correctly identifies them as structurally coupled, but the module has no single purpose. The engineering force is **coupling by proximity, not by intent**: functions that happen to share a dependency get clustered together regardless of what they do.

### Root Causes

**1. Accidental coupling through shared state.**
Multiple unrelated features depend on the same data structure, forcing them into the same structural cluster.
- *Confirm:* Semantic sub-clusters correspond to different domain concepts. The shared dependency is a data structure (struct, class, database table).
- *Action:* Split the module along semantic sub-cluster boundaries. Extract the shared state into its own module.

**2. Catch-all module.**
The module is a dumping ground for functions that didn't have an obvious home.
- *Confirm:* Module name is generic (`utils`, `helpers`, `misc`, `common`). Semantic sub-clusters are numerous and small.
- *Action:* Distribute the functions to their semantic homes. Dissolve the catch-all module.

**3. Historical accumulation.**
The module was coherent at one point but accumulated unrelated responsibilities over time.
- *Confirm:* Semantic sub-clusters have different authorship periods in git history. The module grew steadily without splits.
- *Action:* Split along semantic sub-cluster boundaries.

### Detection

```
Input: semantic embeddings S (768d), spectral module assignments

For each module M with |M| >= 5:
  S_M = semantic embeddings of M's members
  Compute pairwise cosine similarity matrix within M
  mean_intra_sim = mean of upper triangle of similarity matrix

  // Detect semantic sub-clusters
  Run k-means on S_M for k = 2..min(|M|/3, 6)
  Pick k* that maximizes silhouette score on semantic embeddings
  If k* >= 2 AND silhouette(k*) > 0.3:
    sub_clusters = k-means assignments at k*
    inter_cluster_sim = mean cosine similarity between sub-cluster centroids
    If inter_cluster_sim < 0.3:
      Emit incoherent-module(M, k*, sub_clusters, mean_intra_sim, inter_cluster_sim)
```

### Severity Model

```
severity = 0.4 * scatter_factor + 0.3 * size_factor + 0.3 * sub_cluster_factor

scatter_factor      = 1.0 - clamp(mean_intra_sim / 0.5, 0, 1)
size_factor         = clamp(|M| / 30, 0, 1)
sub_cluster_factor  = clamp(k* / 5, 0, 1)

Multipliers:
  Module is the largest module (>30% of nodes): x1.3
  Module name matches utils/helpers/common: x0.7 (expected to be diverse)
  >40% of members independently triggered misplaced-concern: x1.4
  Structural cohesion (spectral) is high despite semantic scatter: x1.2
    (strong accidental coupling)
```

### Example Output

```
incoherent-module
  Symptom: Module core (18 nodes) is structurally coupled but splits into
    3 semantically distinct sub-clusters.
  Impact: The module has no single purpose. Developers cannot predict what
    belongs here. Changes to one sub-cluster risk unintended coupling to
    unrelated concerns.
  Investigate: The 3 sub-clusters appear to be:
    (1) auth: validate_token, check_permissions, session_lookup (6 nodes)
    (2) config: load_settings, parse_env, defaults (5 nodes)
    (3) logging: format_log, write_audit, trace_request (7 nodes)
    Should these be separate modules?
  Evidence:
    Module size: 18 nodes
    Semantic sub-clusters: 3 (silhouette: 0.52)
    Mean intra-module similarity: 0.19
    Inter-sub-cluster similarity: 0.14
    Structural cohesion (spectral): 0.71
```

### False Positive Suppression

- **Small modules:** Suppress if |M| < 5 (too few nodes for meaningful semantic clustering).
- **Utility modules:** If module name matches `utils`/`helpers`/`common`/`shared`/`lib`, demote severity by 0.5x (semantic diversity is expected).
- **Low embedding coverage:** If >30% of module members lack semantic embeddings (source text too short), suppress.
- **Weak sub-clustering:** If the best silhouette score across all k is <0.3, suppress (no clear semantic sub-structure).

### Interactions

- **`misplaced-concern`:** When >40% of a module's members trigger misplaced-concern, suppress the individual misplaced-concern diagnostics and fire incoherent-module. This is a suppression rule, not a co-fire.
- **`cross-package-coupling`:** An incoherent module that spans packages is a compound problem. The structural coupling crosses both semantic and organizational boundaries.
- **`wide-interface`:** Incoherent modules often have wide interfaces because each semantic sub-cluster couples to different external modules.

---

## Diagnostic 10: `shadow-dependency`

**Phase:** 2 | **Absorbs:** — | **Replaced by:** — | **Flag:** `--experimental`

### First-Principles Justification

Two nodes in different modules may do the same thing independently. They are semantically near-identical but have no structural link -- no shared caller, no shared import, no edge between them. This indicates a missing abstraction: the same logic was implemented twice because neither developer knew the other's code existed. The engineering force is **discovery failure at scale**: in a large codebase, finding existing implementations is harder than writing a new one.

### Root Causes

**1. Independent implementation of the same requirement.**
Two teams needed the same logic and each wrote their own version.
- *Confirm:* Nodes have high semantic similarity (>0.85). No structural path of length <=3 between them. Different authors in git history.
- *Action:* Extract shared logic into a common module. Both callers depend on it.

**2. Copy-paste divergence.**
One implementation was copied from the other and has since diverged slightly.
- *Confirm:* Semantic similarity is very high (>0.9). Code structure is nearly identical. Git history may show the copy event.
- *Action:* Consolidate back into a single implementation.

**3. Coincidental similarity.**
Two functions happen to have similar embeddings because they use similar vocabulary (e.g., two different validators that both check "email" and "format") but serve different purposes.
- *Confirm:* Despite high embedding similarity, the functions operate on different types, in different contexts, with different callers.
- *Action:* Not a problem. Suppress.

### Detection

```
Input: semantic embeddings S (768d), module assignments, graph G

// O(n^2) -- gated behind --experimental flag
For each pair (u, v) where module(u) != module(v):
  sim = cosine_similarity(S[u], S[v])
  If sim > 0.85:
    structural_distance = shortest_path_length(G, u, v)  // BFS, cap at 4
    If structural_distance > 3 OR no path exists:
      // Check for shared domain references (type names, imports)
      shared_refs = |referenced_types(u) ∩ referenced_types(v)|
      If shared_refs >= 1:
        Emit shadow-dependency(u, v, sim, structural_distance, shared_refs)
```

### Severity Model

```
severity = 0.4 * similarity_factor + 0.3 * distance_factor + 0.3 * ref_factor

similarity_factor = clamp((sim - 0.85) / 0.15, 0, 1)
distance_factor   = 1.0 if no path exists, else clamp(structural_distance / 5, 0, 1)
ref_factor        = clamp(shared_refs / 3, 0, 1)

Multipliers:
  Both nodes have >5 callers each: x1.3 (duplicated logic is widely used)
  Nodes are in the same declared package: x1.2 (should have found each other)
  One node was created >6 months after the other (git): x1.2 (the second is likely a re-invention)
```

### Example Output

```
shadow-dependency [experimental]
  Symptom: auth::validate_email and user::check_email_format are semantically
    similar (0.92) but have no structural link.
  Impact: The same validation logic exists in two places. Bug fixes applied
    to one will not propagate to the other. Behavior may silently diverge.
  Investigate: Are these truly the same logic? If so, extract into a shared
    validation module. If they serve subtly different purposes, rename to
    make the distinction explicit.
  Evidence:
    Semantic similarity: 0.92
    Structural distance: no path within 4 hops
    Shared type references: EmailAddress, ValidationError
    Callers of auth::validate_email: 8
    Callers of user::check_email_format: 5
```

### False Positive Suppression

- **Trait implementations:** If both nodes implement the same trait/interface (detected by `inherits` edges to the same target), suppress. Implementations of the same interface are expected to be similar.
- **Test doubles:** If either node is in a test path, suppress.
- **Generic vocabulary:** If shared_refs are all from standard library types (String, Vec, Option, Result), suppress (coincidental similarity).
- **Small functions:** If either node's source text is <30 tokens, suppress (too short for meaningful semantic comparison).

### Interactions

- **`misplaced-concern`:** If one of the shadow-dependency pair also triggers misplaced-concern pointing at the other's module, the evidence compounds: the node is both semantically similar to and semantically belonging in the same place. High confidence.
- **`incoherent-module`:** Shadow dependencies across an incoherent module's sub-clusters may indicate the module should be restructured to consolidate the duplicated logic.

---

## Diagnostic 11: `scattered-api`

**Phase:** 2 | **Absorbs:** — | **Replaced by:** —

### First-Principles Justification

A module's public surface should be intentionally narrow. When many entry points are semantically redundant -- they do roughly the same thing but with slightly different signatures or contexts -- the API is scattered. Callers don't know which entry point to use, behavior is inconsistent, and maintenance burden multiplies. The engineering force is **accretion without curation**: each new feature adds a new entry point instead of extending an existing one because the developer didn't find (or didn't trust) the existing one.

### Root Causes

**1. Multiple authors, no API owner.**
Different developers each added their own entry point for similar functionality.
- *Confirm:* Redundant entry points have different authors in git history. Naming conventions are inconsistent.
- *Action:* Consolidate into a single entry point with clear parameters. Deprecate the others.

**2. Version accumulation.**
The API evolved through additive changes. `process_v1`, `process_v2`, `process_with_options` all coexist.
- *Confirm:* Entry point names follow version/suffix patterns. Older versions have declining caller counts.
- *Action:* Deprecate old versions. Migrate callers to the current version.

**3. Intentional overloading.**
The language doesn't support function overloading, so multiple functions with different signatures serve the same purpose.
- *Confirm:* Entry points have the same stem name with different suffixes (`parse_str`, `parse_file`, `parse_bytes`).
- *Action:* Not necessarily a problem. Suppress if the naming convention is consistent.

### Detection

```
Input: semantic embeddings S (768d), graph G, module assignments, role assignments

For each module M:
  entry_points = {v in M : role(v) == "entry_point" OR in_degree_from_outside(v) > 0}
  If |entry_points| < 3: skip

  // Check for semantic redundancy among entry points
  S_E = semantic embeddings of entry_points
  pairwise_sim = cosine similarity matrix of S_E
  redundant_pairs = {(u, v) : pairwise_sim[u][v] > 0.7 AND u != v}

  // Check for convergent callees (entry points that call the same internal functions)
  For each pair in redundant_pairs:
    callees_u = set of direct callees of u within M
    callees_v = set of direct callees of v within M
    callee_overlap = |callees_u ∩ callees_v| / |callees_u ∪ callees_v|  // Jaccard

  redundant_cluster = connected components of redundant_pairs where callee_overlap > 0.5
  For each cluster with |cluster| >= 3:
    Emit scattered-api(M, cluster, mean_pairwise_sim, mean_callee_overlap)
```

### Severity Model

```
severity = 0.4 * cluster_size_factor + 0.3 * redundancy_factor + 0.3 * overlap_factor

cluster_size_factor = clamp((|cluster| - 2) / 5, 0, 1)
redundancy_factor   = clamp(mean_pairwise_sim / 0.9, 0, 1)
overlap_factor      = mean_callee_overlap

Multipliers:
  Module is a declared public API (has exports or is referenced by >50% of other modules): x1.3
  Cluster entry points have inconsistent naming patterns: x1.2
  Cluster entry points have identical return types: x1.1
```

### Example Output

```
scattered-api
  Symptom: Module http has 5 semantically redundant entry points that
    converge on the same internal callees.
  Impact: Callers see 5 ways to do roughly the same thing. Behavior may
    be inconsistent across entry points. Maintenance cost scales with
    the number of redundant paths.
  Investigate: Can these be consolidated into a single entry point with
    an options/config parameter? Or are the differences intentional?
  Evidence:
    Redundant entry points: send_request, make_request, do_request,
      execute_request, fire_request
    Mean pairwise similarity: 0.83
    Shared callees: build_headers, serialize_body, open_connection
    Mean callee overlap (Jaccard): 0.72
```

### False Positive Suppression

- **Intentional overloads:** If all entry points in the cluster share a common stem and differ only by a type suffix (`_str`, `_bytes`, `_file`, `_async`), suppress.
- **Trait implementations:** If the entry points are implementations of different traits/interfaces, suppress (they serve different contracts).
- **Low callee overlap:** If mean callee_overlap < 0.3, suppress (similar names but different internals = different functions).
- **Small cluster:** Suppress if |cluster| < 3.

### Interactions

- **`wide-interface`:** A scattered API often manifests as a wide interface from the caller's perspective. If the calling module triggers wide-interface on this module, and the coupling points overlap with the scattered-api cluster, consolidating the API will also narrow the interface.
- **`overloaded-utility`:** If one entry point in the cluster is also an overloaded-utility, it is likely the "real" function and the others are wrappers. Consolidating onto that function is the natural action.

---

## Diagnostic 12: `misplaced-concern` (upgraded)

**Phase:** 3 | **Absorbs:** Phase 2 `misplaced-concern` | **Replaced by:** —

### First-Principles Justification

Same as Phase 2's misplaced-concern: a node sits in a module that doesn't match its purpose. The difference is detection power. Phase 2 uses cosine distance to a module centroid, which is a single-hop, single-layer, per-codebase signal. Phase 3 uses R-GIN reconstruction error: the model learns to reconstruct a node's semantic embedding from its structural neighborhood. When the reconstruction fails, the node's structural position is inconsistent with its content. This is multi-hop (2-layer GIN), cross-layer (calls/imports/inherits processed separately), role-aware (spectral PEs encode structural role), and cross-repo calibrated (trained on 500+ repos).

### Root Causes

Same as Phase 2 `misplaced-concern`. The upgrade is in detection, not in the causes.

### Detection

```
Input: R-GIN model (trained), semantic embeddings S, graph G

For each node v:
  z_str(v) = R-GIN forward pass on G, producing 160d structural embedding
  s_hat(v) = reconstruction_head(z_str(v))  // 160d -> 768d projection
  reconstruction_error(v) = 1.0 - cosine_similarity(s_hat(v), S[v])

  // Calibrated threshold from training distribution
  threshold = mean(reconstruction_error) + 2 * std(reconstruction_error)
  // Per-role threshold adjustment
  role_baseline = mean reconstruction error for nodes with same role across training set
  adjusted_threshold = max(threshold, role_baseline + 1.5 * std_role)

If reconstruction_error(v) > adjusted_threshold:
  // Identify suggested module by finding which module's structural
  // neighborhood would best reconstruct v's semantic embedding
  For each module M' != module(v):
    counterfactual_error = reconstruction_error if v were in M'
  best_module = argmin counterfactual_error
  Emit misplaced-concern(v, module(v), best_module, reconstruction_error(v), adjusted_threshold)
```

### Severity Model

```
severity = 0.5 * error_factor + 0.3 * confidence_factor + 0.2 * counterfactual_factor

error_factor         = clamp((reconstruction_error - adjusted_threshold) /
                              (3 * std(reconstruction_error)), 0, 1)
confidence_factor    = 1.0 - (reconstruction_error_rank / |nodes|)
                       (how extreme this error is relative to all nodes)
counterfactual_factor = clamp((reconstruction_error - counterfactual_error_best) /
                               reconstruction_error, 0, 1)
                       (how much better the alternative module would be)

Multipliers:
  Phase 2 misplaced-concern also fired on this node: x1.2 (both methods agree)
  Node has >3 structural edges to best_module: x1.3
  >40% of module triggers: suppress individuals, fire incoherent-module instead
```

### Example Output

```
misplaced-concern
  Symptom: billing::validate_jwt_token has high structural-semantic
    reconstruction error (0.74, threshold: 0.42).
  Impact: The R-GIN model cannot reconstruct this node's semantic content
    from its structural neighborhood. Its purpose (JWT validation) is
    inconsistent with its structural position in billing.
  Investigate: This node reconstructs 3.2x better in module auth.
    Check whether moving it would break billing's internal coupling
    or whether billing only calls it through a single entry point.
  Evidence:
    Reconstruction error: 0.74 (threshold: 0.42, role-adjusted: 0.38)
    Best alternative module: auth (counterfactual error: 0.23)
    Error improvement: 69%
    Structural edges to billing: 2, to auth: 4
```

### False Positive Suppression

Same as Phase 2 `misplaced-concern`, plus:
- **Low reconstruction error variance:** If std(reconstruction_error) < 0.05 across all nodes, the model is not discriminating well. Suppress all misplaced-concern diagnostics and log a warning.
- **Model confidence:** If the R-GIN was trained on <100 repos, demote all severities by 0.5x (insufficient calibration).

### Interactions

- **Phase 2 suppression:** When Phase 3 `misplaced-concern` is active, all Phase 2 `misplaced-concern` diagnostics are suppressed.
- **`incoherent-module`:** Same >40% suppression rule as Phase 2.
- **`coupling-mismatch`:** If a node triggers both misplaced-concern and coupling-mismatch, the structural evidence is very strong: the node is wrong both in its overall position and in its per-layer role profile.

---

## Diagnostic 13: `coupling-mismatch`

**Phase:** 3 | **Absorbs:** `layer-discrepancy` (never implemented at Phase 1) | **Replaced by:** —

### First-Principles Justification

A node can play different structural roles in different dependency layers. It may be a hub in the call graph (everything calls it) but a leaf in the import graph (it imports nothing). Small discrepancies are normal -- a utility function is naturally a call-hub and an import-leaf. Large discrepancies indicate structural tension: the node's position in one layer is inconsistent with its position in another. The engineering force is **cross-layer architectural drift**: the call graph evolves independently of the import graph, and nobody checks whether they still agree on each node's role.

### Root Causes

**1. Import-call inversion.**
A node is heavily imported (types, constants) but rarely called. Or heavily called but its module is rarely imported.
- *Confirm:* z_calls and z_imports point in very different directions. High in-degree in one layer, low in the other.
- *Action:* Check whether the imports are type-only (benign) or whether the node's callable interface is under-used despite being widely imported.

**2. Inheritance-call mismatch.**
A node is central in the type hierarchy (many implementations) but peripheral in the call graph (nobody actually calls the trait methods on these implementations).
- *Confirm:* z_inherits shows high centrality; z_calls shows low centrality. Implementations exist but are unused at runtime.
- *Action:* Check for dead implementations. The type hierarchy may be over-designed relative to actual usage.

**3. Transitional state.**
A refactoring is in progress. The node's call-graph role has changed but its import-graph role hasn't caught up (or vice versa).
- *Confirm:* Recent git history shows significant changes to the node's callers or importers. The mismatch is recent.
- *Action:* Complete the refactoring. The mismatch is a symptom of incomplete migration.

### Detection

```
Input: R-GIN per-layer embeddings (z_calls 32d, z_imports 32d, z_inherits 32d)

For each node v:
  // Compute pairwise disagreement between per-layer embeddings
  For each layer pair (r1, r2) in {calls, imports, inherits}:
    // Normalize embeddings to unit vectors
    z_r1_norm = z_r1(v) / ||z_r1(v)||
    z_r2_norm = z_r2(v) / ||z_r2(v)||
    disagreement(r1, r2) = 1.0 - cosine_similarity(z_r1_norm, z_r2_norm)

  max_disagreement = max over all layer pairs
  mean_disagreement = mean over all layer pairs

  // Calibrated threshold from training distribution
  threshold = mean(max_disagreement_all_nodes) + 2 * std(max_disagreement_all_nodes)

If max_disagreement(v) > threshold:
  worst_pair = argmax disagreement
  Emit coupling-mismatch(v, worst_pair, max_disagreement, mean_disagreement)
```

### Severity Model

```
severity = 0.4 * disagreement_factor + 0.3 * impact_factor + 0.3 * consistency_factor

disagreement_factor = clamp((max_disagreement - threshold) /
                             (3 * std(max_disagreement)), 0, 1)
impact_factor       = max(centrality_r1, centrality_r2)
                      (how important the node is in its most central layer)
consistency_factor  = mean_disagreement / max_disagreement
                      (1.0 if all pairs disagree equally = systematic mismatch,
                       low if only one pair disagrees = localized mismatch)

Multipliers:
  Node is an entry point: x0.7 (entry points legitimately have different roles per layer)
  Node is in a cycle: x1.2 (cycles distort per-layer roles)
  Worst pair includes inherits AND |inherits-edges| < 3: x0.7 (sparse inheritance data)
```

### Example Output

```
coupling-mismatch
  Symptom: core::EventBus has a coupling mismatch between its call-graph role
    and its import-graph role (disagreement: 0.81, threshold: 0.52).
  Impact: EventBus is a hub in the import graph (imported by 12 modules for its
    types) but peripheral in the call graph (called by only 2 modules at runtime).
    The type hierarchy promises an interface that the runtime barely uses.
  Investigate: Are the 10 modules that import EventBus types but never call
    EventBus methods using dead abstractions? Or is EventBus meant to be called
    indirectly (via dynamic dispatch) and the call graph is incomplete?
  Evidence:
    Call-graph centrality: p15
    Import-graph centrality: p89
    Disagreement (calls vs imports): 0.81 (threshold: 0.52)
    Disagreement (calls vs inherits): 0.44
    Disagreement (imports vs inherits): 0.39
```

### False Positive Suppression

- **Type-only nodes:** If the node is a struct/enum/type with no callable methods, suppress. Type-only nodes are naturally import-central and call-peripheral.
- **Sparse layers:** If the node has <2 edges in the layer with lower centrality, suppress (insufficient data for that layer).
- **Entry points:** If role is "entry_point", raise the threshold by 1.5x (entry points have legitimately different per-layer profiles).
- **Standard library wrappers:** If the node re-exports or wraps a standard library type, suppress.

### Interactions

- **`misplaced-concern` (Phase 3):** A node that triggers both coupling-mismatch and misplaced-concern has very strong structural evidence of a problem. The coupling-mismatch says the per-layer roles disagree; the misplaced-concern says the overall position is wrong. Both fire independently.
- **`overloaded-utility`:** An overloaded-utility with a coupling mismatch may be overloaded in one layer (calls) but not another (imports), suggesting the overload is from runtime coupling, not compile-time coupling. This helps the developer decide how to split it.
- **`layer-violation`:** Coupling mismatches often co-occur with layer violations because the per-layer role disagreement manifests as edges in unexpected directions.

---

## What Was Removed and Why

These former diagnostics were evaluated and cut:

| Former Diagnostic | Decision | Rationale |
|---|---|---|
| `spectral-outlier` | Demoted to internal signal | ~40% FP rate. Z-score from spectral centroid without semantic confirmation is noise. Feeds Phase 2's `misplaced-concern` as input, not as a user-facing diagnostic. |
| `god-module` | Subsumed by `incoherent-module` | "Big module" without semantic coherence check fires on every monolith. `incoherent-module` adds the semantic sub-cluster check that distinguishes legitimate large modules from incoherent ones. |
| `low-cohesion` | Subsumed by `incoherent-module` | Spectral cohesion/separation ratio is too abstract for developers. Demoted to a health metric. `incoherent-module` provides actionable sub-cluster information instead of a ratio. |
| `orphan` | Removed | Zero-edge nodes are detectable by every language's dead-code linter (`rustc` warnings, ESLint `no-unused-vars`, etc.). Not structural intelligence. |
| `phantom-import` | Removed | >50% FP rate from type-only imports, re-exports, and macro-generated code. Unused import detection is a solved problem in language-specific linters. |
| `layer-discrepancy` | Replaced by `coupling-mismatch` | Cross-layer centrality gap from raw degree percentiles is too noisy without learned embeddings. Never implemented at Phase 1. Implemented only at Phase 3 with R-GIN per-layer decomposition. |
| `reverse-dependency` | Absorbed into `layer-violation` | Bidirectional package flow is a special case of layer violation. `layer-violation` is more precise: individual edges, inferred direction, role-based violations. |
| `bidirectional-dependency` | Absorbed into `layer-violation` | Bidirectional spectral module boundary edges. Same rationale as `reverse-dependency`. |
| `module-separation:weak` | Subsumed by `incoherent-module` | "Biggest module is big" is a shallow signal. Subsumed first by `god-module`, which is itself subsumed by `incoherent-module`. |
| `fragile-hub` | Absorbed into `overloaded-utility` | High betweenness without directionality fires on bridges, which are structurally expected. `overloaded-utility` adds direction score and caller diversity, suppressing bridges. |
| `import-call-leakage` | Removed | Proposed for Phase 3. Duplicates linter functionality. Massive FP on type-heavy codebases (Rust, TypeScript). The cross-layer prediction is a training metric, not a user diagnostic. |
| `semantic-structural-drift` | Demoted to health metric | Not actionable per-node. Tells you "disagreement lives at scale X" but not "fix node Y." Emitted as Rayleigh quotient in the health JSON. |

---

## Health Metrics

These are emitted in the JSON `health` section as numeric fields. They track codebase health over time but are not actionable per-node findings:

| Metric | Field | Description |
|---|---|---|
| Spectral coverage ratio | `spectral_coverage_ratio` | Fraction of nodes that received spectral fingerprints. Low values indicate disconnected components limiting analysis quality. |
| Self-edge drop ratio | `self_edge_drop_ratio` | Fraction of scoped edges that collapsed into self-edges at the current analysis level. High values suggest `--level symbol` would be more informative. |
| Rayleigh quotient | `rayleigh_quotient` | Semantic-structural drift. Measures how much semantic variation aligns with structural variation. Low values indicate the architecture matches intent; high values indicate systematic disagreement. |
| Largest module ratio | `largest_module_ratio` | Fraction of all nodes in the largest spectral module. Values >0.5 indicate weak overall modularity. |
| Module size distribution | `module_size_gini` | Gini coefficient of module sizes. 0.0 = all modules equal size. 1.0 = one module has everything. Replaces god-module as a tracking metric. |
| Module edge-share distribution | `module_edge_share_gini` | Gini coefficient of how edges distribute across modules. Complements module_size_gini. |
| Per-module spectral cohesion | `module_cohesion[]` | Array of cohesion/separation ratios per module. Replaces low-cohesion as a tracking metric. |

---

## Suppression Rules

These rules prevent redundant or misleading diagnostic combinations:

1. **Phase 3 `misplaced-concern` active --> suppress Phase 2 `misplaced-concern`.** The Phase 3 version uses R-GIN reconstruction error, which is strictly more informative than centroid distance. Both firing on the same node would be redundant.

2. **`cycle-member` and `layer-violation` on the same module pair --> suppress `layer-violation`.** If two modules are in the same SCC, the cycle is the fundamental problem. Layer violations within a cycle are symptoms, not root causes.

3. **>40% of a module's members trigger `misplaced-concern` --> suppress individual `misplaced-concern`, fire `incoherent-module`.** When most of a module is misplaced, the module boundary itself is wrong. Individual "move this node" advice is less useful than "split this module."

4. **Phase 3 `coupling-mismatch` has no Phase 1 equivalent to suppress.** `layer-discrepancy` was never implemented. No suppression rule needed.

5. **`overloaded-utility` and `fragile-hub` (legacy) --> only `overloaded-utility` fires.** If old `fragile-hub` diagnostics exist in cached results, overloaded-utility takes precedence.
