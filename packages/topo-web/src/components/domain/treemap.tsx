import { useMemo, useState, useRef, useEffect, Fragment } from "react";
import * as d3 from "d3";
import type {
  DomainNode,
  DomainDependency,
  Issue,
  GraphNode,
} from "../../lib/types";
import { cn, healthBg, healthColor, domainColor } from "../../lib/utils";
import {
  ChevronRight,
  ChevronLeft,
  AlertTriangle,
  ArrowUpRight,
} from "lucide-react";

interface TreemapProps {
  domain: DomainNode;
  issues: Issue[];
  graphNodes: GraphNode[];
  onSelect: (domain: DomainNode) => void;
  selectedPath: string | null;
}

interface ExternalDep {
  target: string;
  weight: number;
  sources: string[];
}

export function DomainTreemap({
  domain,
  issues,
  graphNodes,
  onSelect,
  selectedPath,
}: TreemapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [zoomStack, setZoomStack] = useState<DomainNode[]>([domain]);
  const [hoveredPath, setHoveredPath] = useState<string | null>(null);
  const [hoveredExternal, setHoveredExternal] = useState<string | null>(null);

  const currentDomain = zoomStack[zoomStack.length - 1];

  // Reset zoom when data changes.
  useEffect(() => {
    setZoomStack([domain]);
  }, [domain]);

  // Track container size.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Count issues per domain path (including parent roll-up).
  const issuesByDomain = useMemo(() => {
    const map = new Map<string, number>();
    for (const issue of issues) {
      for (const anchor of issue.anchors) {
        const node = graphNodes.find((n) => n.id === anchor.node_id);
        if (!node) continue;
        const path = node.domain_path;
        map.set(path, (map.get(path) ?? 0) + 1);
        const parts = path.split("/");
        for (let i = 1; i < parts.length; i++) {
          const parent = parts.slice(0, i).join("/");
          map.set(parent, (map.get(parent) ?? 0) + 1);
        }
      }
    }
    return map;
  }, [issues, graphNodes]);

  // ── External dependency data (for gutter when zoomed in) ──

  const scope = currentDomain.path;

  const externalDeps = useMemo((): ExternalDep[] => {
    if (zoomStack.length <= 1) return [];
    const map = new Map<string, { weight: number; sources: Set<string> }>();
    for (const child of currentDomain.children) {
      for (const dep of child.dependencies) {
        if (
          !dep.target_path.startsWith(scope + "/") &&
          dep.target_path !== scope
        ) {
          const entry = map.get(dep.target_path);
          if (entry) {
            entry.weight += dep.weight;
            entry.sources.add(child.path);
          } else {
            map.set(dep.target_path, {
              weight: dep.weight,
              sources: new Set([child.path]),
            });
          }
        }
      }
    }
    return [...map.entries()]
      .map(([target, data]) => ({
        target,
        weight: data.weight,
        sources: [...data.sources],
      }))
      .sort((a, b) => b.weight - a.weight);
  }, [currentDomain, zoomStack, scope]);

  // Which cells depend on a given external target?
  const cellsForExternal = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const dep of externalDeps) {
      map.set(dep.target, new Set(dep.sources));
    }
    return map;
  }, [externalDeps]);

  // Which external targets does a given cell depend on?
  const externalsForCell = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const child of currentDomain.children) {
      const targets = new Set<string>();
      for (const dep of child.dependencies) {
        if (
          !dep.target_path.startsWith(scope + "/") &&
          dep.target_path !== scope
        ) {
          targets.add(dep.target_path);
        }
      }
      if (targets.size > 0) map.set(child.path, targets);
    }
    return map;
  }, [currentDomain, scope]);

  // ── Treemap layout ──

  const layout = useMemo(() => {
    if (!size.width || !size.height || !currentDomain.children.length)
      return [];

    const hierarchy = d3
      .hierarchy(currentDomain)
      .sum((d) => (d.children.length === 0 ? d.size : 0))
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));

    const root = d3
      .treemap<DomainNode>()
      .size([size.width, size.height])
      .paddingTop(28)
      .paddingRight(3)
      .paddingBottom(3)
      .paddingLeft(3)
      .paddingInner(3)
      .round(true)
      .tile(d3.treemapSquarify)(hierarchy);

    return root.descendants().slice(1);
  }, [currentDomain, size]);

  // ── Interaction helpers ──

  function zoomIn(child: DomainNode) {
    if (child.children.length > 0) {
      setZoomStack([...zoomStack, child]);
    }
    onSelect(child);
  }

  function zoomTo(index: number) {
    setZoomStack(zoomStack.slice(0, index + 1));
    setHoveredExternal(null);
  }

  // Gutter hover → cell highlighting
  function cellDimState(path: string): "normal" | "highlighted" | "dimmed" {
    if (!hoveredExternal) return "normal";
    return cellsForExternal.get(hoveredExternal)?.has(path)
      ? "highlighted"
      : "dimmed";
  }

  // Cell hover → gutter chip highlighting
  function chipDimState(target: string): "normal" | "highlighted" | "dimmed" {
    if (!hoveredPath) return "normal";
    return externalsForCell.get(hoveredPath)?.has(target)
      ? "highlighted"
      : "dimmed";
  }

  return (
    <div className="flex flex-col h-full">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1 px-4 py-2.5 border-b border-border bg-surface/50 text-sm shrink-0">
        {zoomStack.map((d, i) => (
          <Fragment key={d.path}>
            {i > 0 && <ChevronRight className="w-3 h-3 text-muted" />}
            <button
              onClick={() => zoomTo(i)}
              className={cn(
                "px-1.5 py-0.5 rounded transition-colors",
                i === zoomStack.length - 1
                  ? "text-foreground font-medium"
                  : "text-muted hover:text-foreground",
              )}
            >
              {d.label}
            </button>
          </Fragment>
        ))}

        {zoomStack.length > 1 && (
          <button
            onClick={() => zoomTo(zoomStack.length - 2)}
            className="ml-auto text-muted hover:text-foreground transition-colors flex items-center gap-1 text-xs"
          >
            <ChevronLeft className="w-3 h-3" />
            Back
          </button>
        )}
      </div>

      {/* Cross-cutting strip */}
      {currentDomain.cross_cutting.length > 0 && (
        <div className="flex items-center gap-2 px-4 py-1.5 border-b border-border bg-surface/30 text-xs shrink-0">
          <span className="text-muted font-medium">Cross-cutting:</span>
          {currentDomain.cross_cutting.map((cc) => (
            <span
              key={cc.node_id}
              className="px-2 py-0.5 rounded bg-purple/10 text-purple border border-purple/20 font-mono text-[11px]"
            >
              {cc.node_id}
            </span>
          ))}
        </div>
      )}

      {/* ── External dependency gutter (when zoomed in) ── */}
      {externalDeps.length > 0 && (
        <div className="flex items-center gap-2 px-4 py-1.5 border-b border-border bg-canvas text-xs shrink-0 overflow-x-auto">
          <ArrowUpRight className="w-3 h-3 text-muted shrink-0" />
          <span className="text-muted font-medium shrink-0">External:</span>
          {externalDeps.map((dep) => {
            const state = chipDimState(dep.target);
            return (
              <button
                key={dep.target}
                className={cn(
                  "flex items-center gap-1.5 px-2 py-0.5 rounded-md border transition-all duration-150 shrink-0",
                  hoveredExternal === dep.target
                    ? "border-accent/50 bg-accent/10"
                    : state === "dimmed"
                      ? "border-border/30 opacity-30"
                      : "border-border/50 hover:border-accent/30",
                )}
                style={{
                  borderLeftWidth: 3,
                  borderLeftColor: domainColor(dep.target),
                }}
                onMouseEnter={() => setHoveredExternal(dep.target)}
                onMouseLeave={() => setHoveredExternal(null)}
              >
                <span className="font-mono text-[11px] text-foreground">
                  {dep.target}
                </span>
                <span className="text-[10px] text-muted tabular-nums">
                  {dep.weight}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* ── Treemap canvas ── */}
      <div ref={containerRef} className="flex-1 relative overflow-hidden">
        {layout.map((node) => {
          const d = node.data;
          const isLeaf = !node.children || node.children.length === 0;
          const w = node.x1 - node.x0;
          const h = node.y1 - node.y0;
          const tooSmall = w < 50 || h < 24;
          const issueCount = issuesByDomain.get(d.path) ?? 0;
          const isSelected = selectedPath === d.path;
          const dim = cellDimState(d.path);

          return (
            <div
              key={d.path}
              className={cn(
                "absolute overflow-hidden transition-all duration-200",
                isLeaf
                  ? "rounded-md border cursor-pointer hover:brightness-125 hover:border-accent/40"
                  : "rounded-lg",
                isSelected
                  ? "ring-2 ring-accent ring-offset-1 ring-offset-canvas z-10"
                  : isLeaf
                    ? "border-border/30"
                    : "",
                dim === "dimmed" && "opacity-35",
                dim === "highlighted" && "ring-1 ring-accent/50",
              )}
              style={{
                left: node.x0,
                top: node.y0,
                width: w,
                height: h,
                backgroundColor: isLeaf
                  ? healthBg(d.health?.topo_health_score)
                  : "rgba(22, 27, 34, 0.4)",
              }}
              onClick={(e) => {
                e.stopPropagation();
                zoomIn(d);
              }}
              onMouseEnter={() => setHoveredPath(d.path)}
              onMouseLeave={() => setHoveredPath(null)}
            >
              {!tooSmall && (
                <div
                  className={cn(
                    "flex items-center gap-1.5 px-2 py-1 truncate",
                    isLeaf ? "" : "border-b border-border/20",
                  )}
                >
                  <span
                    className={cn(
                      "truncate",
                      isLeaf
                        ? "text-xs font-medium text-foreground"
                        : "text-[11px] font-semibold text-muted uppercase tracking-wider",
                    )}
                  >
                    {d.label}
                  </span>

                  {d.health && (
                    <span
                      className={cn(
                        "text-[10px] font-mono tabular-nums ml-auto shrink-0",
                        healthColor(d.health.topo_health_score),
                      )}
                    >
                      {Math.round(d.health.topo_health_score * 100)}
                    </span>
                  )}

                  {issueCount > 0 && (
                    <span className="flex items-center gap-0.5 text-warning shrink-0">
                      <AlertTriangle className="w-2.5 h-2.5" />
                      <span className="text-[10px] font-mono">
                        {issueCount}
                      </span>
                    </span>
                  )}
                </div>
              )}

              {/* Member list for leaves with space */}
              {isLeaf && h > 60 && (
                <div className="px-2 py-1">
                  {d.members
                    .slice(0, Math.floor((h - 30) / 16))
                    .map((m) => (
                      <div
                        key={m}
                        className="text-[10px] font-mono text-muted/60 truncate leading-4"
                      >
                        {m.split(".").pop()}
                      </div>
                    ))}
                </div>
              )}

              {/* ── Coupling segments bar ── */}
              {d.dependencies.length > 0 && h > 20 && (
                <CouplingBar
                  deps={d.dependencies}
                  highlightTarget={hoveredExternal}
                />
              )}
            </div>
          );
        })}

        {/* Leaf empty state */}
        {currentDomain.children.length === 0 && (
          <div className="flex items-center justify-center h-full text-muted">
            <div className="text-center">
              <p className="text-sm font-medium mb-1">
                {currentDomain.label}
              </p>
              <p className="text-xs">
                {currentDomain.size} nodes &middot; leaf domain
              </p>
              <div className="mt-3 flex flex-wrap gap-1 justify-center max-w-md">
                {currentDomain.members.map((m) => (
                  <span
                    key={m}
                    className="px-2 py-0.5 text-[11px] font-mono bg-surface rounded border border-border"
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Coupling segments bar ──────────────────────────────────────────

function CouplingBar({
  deps,
  highlightTarget,
}: {
  deps: DomainDependency[];
  highlightTarget: string | null;
}) {
  const total = deps.reduce((s, d) => s + d.weight, 0);
  if (total === 0) return null;

  const segments = deps
    .map((d) => ({
      target: d.target_path,
      pct: (d.weight / total) * 100,
    }))
    .sort((a, b) => b.pct - a.pct);

  return (
    <div className="absolute bottom-0 left-0 right-0 h-[3px] flex">
      {segments.map((seg) => (
        <div
          key={seg.target}
          className="transition-opacity duration-150"
          style={{
            width: `${seg.pct}%`,
            backgroundColor: domainColor(seg.target),
            opacity: highlightTarget
              ? highlightTarget === seg.target
                ? 1
                : 0.15
              : 0.7,
          }}
        />
      ))}
    </div>
  );
}
