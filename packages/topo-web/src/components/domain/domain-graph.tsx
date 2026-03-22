import { useMemo, useState, useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";
import type { DomainNode, Issue, GraphNode } from "../../lib/types";
import { cn, healthColor } from "../../lib/utils";
import { DomainCard } from "./domain-card";
import { ChevronDown, Maximize2, ZoomIn, ZoomOut } from "lucide-react";

// ── Constants ──────────────────────────────────────────────────────

const CARD_W = 200;
const CARD_H = 72;
const CARD_GAP = 20;
const CONTAINER_PAD = 12;
const CONTAINER_HEADER = 28;
const PAD = 48;

// ── Layout types ───────────────────────────────────────────────────

interface NodePos {
  id: string;
  domain: DomainNode;
  x: number;
  y: number;
  w: number;
  h: number;
  isContainer: boolean;
  containerId: string | null;
}

interface EdgeDef {
  id: string;
  source: string;
  target: string;
  weight: number;
}

// ── Graph construction ─────────────────────────────────────────────

function resolveTarget(
  path: string,
  visible: Set<string>,
): string | null {
  if (visible.has(path)) return path;
  const parts = path.split("/");
  for (let i = parts.length - 1; i >= 1; i--) {
    const anc = parts.slice(0, i).join("/");
    if (visible.has(anc)) return anc;
  }
  return null;
}

function buildGraph(root: DomainNode, expanded: Set<string>) {
  const nodes: NodePos[] = [];
  const visible = new Set<string>();

  for (const child of root.children) {
    visible.add(child.path);

    if (expanded.has(child.path) && child.children.length > 0) {
      const numC = child.children.length;
      const cw =
        numC * (CARD_W + CARD_GAP) - CARD_GAP + CONTAINER_PAD * 2;
      const ch = CARD_H + CONTAINER_HEADER + CONTAINER_PAD * 2;

      nodes.push({
        id: child.path,
        domain: child,
        x: 0,
        y: 0,
        w: cw,
        h: ch,
        isContainer: true,
        containerId: null,
      });

      for (const gc of child.children) {
        visible.add(gc.path);
        nodes.push({
          id: gc.path,
          domain: gc,
          x: 0,
          y: 0,
          w: CARD_W,
          h: CARD_H,
          isContainer: false,
          containerId: child.path,
        });
      }
    } else {
      nodes.push({
        id: child.path,
        domain: child,
        x: 0,
        y: 0,
        w: CARD_W,
        h: CARD_H,
        isContainer: false,
        containerId: null,
      });
    }
  }

  // Build edges from leaf-level nodes only.
  const edges: EdgeDef[] = [];
  const seen = new Set<string>();
  const leaves = nodes.filter((n) => !n.isContainer);

  for (const node of leaves) {
    for (const dep of node.domain.dependencies) {
      const target = resolveTarget(dep.target_path, visible);
      if (!target || target === node.id) continue;
      if (node.containerId === target) continue;

      const key = `${node.id}|${target}`;
      if (seen.has(key)) continue;
      seen.add(key);

      edges.push({
        id: key,
        source: node.id,
        target,
        weight: dep.weight,
      });
    }
  }

  return { nodes, edges, visible };
}

// ── Layer assignment (BFS from entry points) ───────────────────────

function computeLayers(
  topIds: string[],
  edges: { source: string; target: string }[],
): Map<string, number> {
  const inDeg = new Map(topIds.map((id) => [id, 0]));
  for (const e of edges) {
    inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
  }

  const layers = new Map<string, number>();
  const queue: string[] = [];

  for (const id of topIds) {
    if ((inDeg.get(id) ?? 0) === 0) {
      layers.set(id, 0);
      queue.push(id);
    }
  }

  let idx = 0;
  while (idx < queue.length) {
    const id = queue[idx++];
    const layer = layers.get(id)!;
    for (const e of edges) {
      if (e.source === id && topIds.includes(e.target)) {
        const prev = layers.get(e.target) ?? -1;
        if (layer + 1 > prev) {
          layers.set(e.target, layer + 1);
        }
        if (!queue.includes(e.target)) {
          queue.push(e.target);
        }
      }
    }
  }

  // Unvisited nodes (cycles) → middle layer.
  const maxL = Math.max(0, ...layers.values());
  for (const id of topIds) {
    if (!layers.has(id)) layers.set(id, Math.ceil(maxL / 2));
  }

  return layers;
}

// ── Edge path (bezier between card centers) ────────────────────────

function edgePath(
  sx: number,
  sy: number,
  sw: number,
  sh: number,
  tx: number,
  ty: number,
  tw: number,
  th: number,
): string {
  const dy = ty - sy;
  const dx = tx - sx;

  let x1: number, y1: number, x2: number, y2: number;

  if (Math.abs(dy) > (sh + th) / 3) {
    // Vertical connection.
    if (dy > 0) {
      x1 = sx;
      y1 = sy + sh / 2;
      x2 = tx;
      y2 = ty - th / 2;
    } else {
      x1 = sx;
      y1 = sy - sh / 2;
      x2 = tx;
      y2 = ty + th / 2;
    }
    const cpY = (y1 + y2) / 2;
    return `M${x1},${y1} C${x1},${cpY} ${x2},${cpY} ${x2},${y2}`;
  } else {
    // Horizontal connection.
    if (dx > 0) {
      x1 = sx + sw / 2;
      y1 = sy;
      x2 = tx - tw / 2;
      y2 = ty;
    } else {
      x1 = sx - sw / 2;
      y1 = sy;
      x2 = tx + tw / 2;
      y2 = ty;
    }
    const cpX = (x1 + x2) / 2;
    return `M${x1},${y1} C${cpX},${y1} ${cpX},${y2} ${x2},${y2}`;
  }
}

// ── Component ──────────────────────────────────────────────────────

interface Props {
  domain: DomainNode;
  issues: Issue[];
  graphNodes: GraphNode[];
  onSelect: (domain: DomainNode) => void;
  selectedPath: string | null;
}

export function DomainGraph({
  domain,
  issues,
  graphNodes,
  onSelect,
  selectedPath,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // ── Pan & Zoom state ──
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const needsFit = useRef(true);

  // Resize observer.
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

  // Build the visible graph from tree + expansion state.
  const { nodes, edges } = useMemo(
    () => buildGraph(domain, expanded),
    [domain, expanded],
  );

  // Issue counts per domain path.
  const issueCounts = useMemo(() => {
    const map = new Map<string, number>();
    for (const issue of issues) {
      for (const anchor of issue.anchors) {
        const gn = graphNodes.find((n) => n.id === anchor.node_id);
        if (!gn) continue;
        const parts = gn.domain_path.split("/");
        for (let i = 1; i <= parts.length; i++) {
          const p = parts.slice(0, i).join("/");
          map.set(p, (map.get(p) ?? 0) + 1);
        }
      }
    }
    return map;
  }, [issues, graphNodes]);

  // ── Force layout ──

  const positioned = useMemo((): NodePos[] => {
    if (!size.width || !size.height || nodes.length === 0) return [];

    const topNodes = nodes.filter((n) => n.containerId === null);
    const childNodes = nodes.filter((n) => n.containerId !== null);

    // Map children → parent container for edge aggregation.
    const childToParent = new Map<string, string>();
    for (const n of childNodes) {
      if (n.containerId) childToParent.set(n.id, n.containerId);
    }

    // Aggregate edges to top-level for layering + force links.
    const tlEdgeMap = new Map<string, number>();
    for (const e of edges) {
      const src = childToParent.get(e.source) ?? e.source;
      const tgt = childToParent.get(e.target) ?? e.target;
      if (src === tgt) continue;
      const key = `${src}|${tgt}`;
      tlEdgeMap.set(key, (tlEdgeMap.get(key) ?? 0) + e.weight);
    }
    const tlEdges = [...tlEdgeMap.entries()].map(([key, weight]) => {
      const [source, target] = key.split("|");
      return { source, target, weight };
    });

    // Layer assignment.
    const topIds = topNodes.map((n) => n.id);
    const layers = computeLayers(topIds, tlEdges);
    const maxLayer = Math.max(0, ...layers.values());
    const bandGap = Math.max(
      120,
      (size.height - PAD * 2) / Math.max(maxLayer + 1, 1),
    );

    // Force simulation.
    interface SimNode extends d3.SimulationNodeDatum {
      id: string;
      w: number;
      h: number;
      targetY: number;
    }

    const simNodes: SimNode[] = topNodes.map((n, i) => ({
      id: n.id,
      w: n.w,
      h: n.h,
      targetY: PAD + (layers.get(n.id) ?? 1) * bandGap + n.h / 2,
      x:
        size.width / 2 +
        (i - topNodes.length / 2) * (CARD_W + 40),
      y: PAD + (layers.get(n.id) ?? 1) * bandGap + n.h / 2,
    }));

    const simNodeIds = new Set(simNodes.map((n) => n.id));
    const simLinks = tlEdges.filter(
      (l) => simNodeIds.has(l.source) && simNodeIds.has(l.target),
    );

    const sim = d3
      .forceSimulation<SimNode>(simNodes)
      .force(
        "y",
        d3.forceY<SimNode>((d) => d.targetY).strength(0.6),
      )
      .force(
        "x",
        d3.forceX<SimNode>(size.width / 2).strength(0.05),
      )
      .force("charge", d3.forceManyBody().strength(-250))
      .force(
        "link",
        d3
          .forceLink(simLinks)
          .id((d: any) => d.id)
          .distance(140)
          .strength(0.15),
      )
      .force(
        "collide",
        d3
          .forceCollide<SimNode>()
          .radius((d) => Math.max(d.w, d.h) / 2 + 16),
      )
      .stop();

    for (let i = 0; i < 300; i++) sim.tick();

    // Collect top-level positions.
    const posMap = new Map<string, { x: number; y: number }>();
    for (const sn of simNodes) {
      posMap.set(sn.id, { x: sn.x!, y: sn.y! });
    }

    const result: NodePos[] = [];

    for (const n of topNodes) {
      const pos = posMap.get(n.id)!;
      result.push({ ...n, x: pos.x, y: pos.y });
    }

    // Position children inside their container.
    for (const n of childNodes) {
      const cPos = posMap.get(n.containerId!)!;
      const container = topNodes.find((tn) => tn.id === n.containerId)!;
      const siblings = childNodes.filter(
        (c) => c.containerId === n.containerId,
      );
      const idx = siblings.indexOf(n);

      const startX =
        cPos.x - container.w / 2 + CONTAINER_PAD + CARD_W / 2;
      const childY =
        cPos.y -
        container.h / 2 +
        CONTAINER_HEADER +
        CONTAINER_PAD +
        CARD_H / 2;

      result.push({
        ...n,
        x: startX + idx * (CARD_W + CARD_GAP),
        y: childY,
      });
    }

    return result;
  }, [nodes, edges, size]);

  // Node position lookup.
  const posById = useMemo(
    () => new Map(positioned.map((n) => [n.id, n])),
    [positioned],
  );

  // Connected set for hover highlighting.
  const connected = useMemo(() => {
    if (!hoveredId) return null;
    const set = new Set([hoveredId]);
    for (const e of edges) {
      if (e.source === hoveredId) set.add(e.target);
      if (e.target === hoveredId) set.add(e.source);
    }
    // Also connect the parent container if hovering a child.
    const hovered = posById.get(hoveredId);
    if (hovered?.containerId) set.add(hovered.containerId);
    return set;
  }, [hoveredId, edges, posById]);

  function toggleExpand(path: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
    needsFit.current = true;
  }

  // ── Pan & Zoom handlers ──

  const fitToView = useCallback(() => {
    if (positioned.length === 0 || !size.width || !size.height) return;
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const n of positioned) {
      minX = Math.min(minX, n.x - n.w / 2);
      maxX = Math.max(maxX, n.x + n.w / 2);
      minY = Math.min(minY, n.y - n.h / 2);
      maxY = Math.max(maxY, n.y + n.h / 2);
    }
    const contentW = maxX - minX;
    const contentH = maxY - minY;
    if (contentW <= 0 || contentH <= 0) return;
    const scaleX = (size.width - PAD * 2) / contentW;
    const scaleY = (size.height - PAD * 2) / contentH;
    const newZoom = Math.min(scaleX, scaleY, 1.5);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    setZoom(newZoom);
    setPan({
      x: size.width / 2 - cx * newZoom,
      y: size.height / 2 - cy * newZoom,
    });
  }, [positioned, size]);

  // Auto fit-to-view on layout changes.
  useEffect(() => {
    if (needsFit.current && positioned.length > 0) {
      needsFit.current = false;
      fitToView();
    }
  }, [positioned, fitToView]);

  function handleMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return;
    // Don't pan if clicking on a card or button.
    if ((e.target as HTMLElement).closest("[data-card], button")) return;
    setIsPanning(true);
    panStart.current = {
      x: e.clientX,
      y: e.clientY,
      panX: pan.x,
      panY: pan.y,
    };
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!isPanning) return;
    setPan({
      x: panStart.current.panX + (e.clientX - panStart.current.x),
      y: panStart.current.panY + (e.clientY - panStart.current.y),
    });
  }

  function handleMouseUp() {
    setIsPanning(false);
  }

  function handleWheel(e: React.WheelEvent) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.92 : 1.08;
    const newZoom = Math.min(Math.max(zoom * factor, 0.2), 3);
    // Zoom centered on cursor.
    const rect = containerRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const scale = newZoom / zoom;
    setPan({
      x: mx - (mx - pan.x) * scale,
      y: my - (my - pan.y) * scale,
    });
    setZoom(newZoom);
  }

  // Separate containers and cards for rendering.
  const containers = positioned.filter(
    (n) => n.isContainer && n.containerId === null,
  );
  const cards = positioned.filter((n) => !n.isContainer);

  return (
    <div className="h-full flex flex-col">
      {/* Cross-cutting strip */}
      {domain.cross_cutting.length > 0 && (
        <div className="flex items-center gap-2 px-4 py-1.5 border-b border-border bg-surface/30 text-xs shrink-0">
          <span className="text-muted font-medium">Cross-cutting:</span>
          {domain.cross_cutting.map((cc) => (
            <span
              key={cc.node_id}
              className="px-2 py-0.5 rounded bg-purple/10 text-purple border border-purple/20 font-mono text-[11px]"
            >
              {cc.node_id}
            </span>
          ))}
        </div>
      )}

      {/* Graph canvas */}
      <div
        ref={containerRef}
        className={cn(
          "flex-1 relative overflow-hidden",
          isPanning ? "cursor-grabbing" : "cursor-grab",
        )}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        {/* Toolbar */}
        <div className="absolute top-3 right-3 z-20 flex items-center gap-1 bg-surface/80 backdrop-blur-sm rounded-lg border border-border p-1">
          <button
            onClick={() => {
              const f = 1.2;
              const newZ = Math.min(zoom * f, 3);
              const s = newZ / zoom;
              setPan({
                x: size.width / 2 - (size.width / 2 - pan.x) * s,
                y: size.height / 2 - (size.height / 2 - pan.y) * s,
              });
              setZoom(newZ);
            }}
            className="p-1.5 rounded text-muted hover:text-foreground hover:bg-surface transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => {
              const f = 0.8;
              const newZ = Math.max(zoom * f, 0.2);
              const s = newZ / zoom;
              setPan({
                x: size.width / 2 - (size.width / 2 - pan.x) * s,
                y: size.height / 2 - (size.height / 2 - pan.y) * s,
              });
              setZoom(newZ);
            }}
            className="p-1.5 rounded text-muted hover:text-foreground hover:bg-surface transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <div className="w-px h-4 bg-border mx-0.5" />
          <button
            onClick={fitToView}
            className="p-1.5 rounded text-muted hover:text-foreground hover:bg-surface transition-colors"
            title="Fit to view"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <span className="text-[10px] text-muted tabular-nums px-1 min-w-[32px] text-center">
            {Math.round(zoom * 100)}%
          </span>
        </div>

        {/* Transform layer */}
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "0 0",
            position: "absolute",
            top: 0,
            left: 0,
          }}
        >
        {/* SVG edge layer */}
        <svg
          className="absolute pointer-events-none"
          width={size.width}
          height={size.height}
          overflow="visible"
          style={{ zIndex: 0 }}
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 6"
              refX="9"
              refY="3"
              markerWidth="8"
              markerHeight="6"
              orient="auto"
            >
              <path
                d="M0,0 L10,3 L0,6"
                fill="rgba(139,148,158,0.5)"
              />
            </marker>
            <marker
              id="arrow-hl"
              viewBox="0 0 10 6"
              refX="9"
              refY="3"
              markerWidth="8"
              markerHeight="6"
              orient="auto"
            >
              <path d="M0,0 L10,3 L0,6" fill="#58a6ff" />
            </marker>
          </defs>

          {edges.map((e) => {
            const src = posById.get(e.source);
            const tgt = posById.get(e.target);
            if (!src || !tgt) return null;

            const isActive =
              hoveredId !== null &&
              (e.source === hoveredId || e.target === hoveredId);
            const isDimmed = connected !== null && !isActive;

            return (
              <path
                key={e.id}
                d={edgePath(
                  src.x,
                  src.y,
                  src.w,
                  src.h,
                  tgt.x,
                  tgt.y,
                  tgt.w,
                  tgt.h,
                )}
                fill="none"
                stroke={isActive ? "#58a6ff" : "#30363d"}
                strokeWidth={
                  isActive
                    ? 2
                    : Math.min(1 + e.weight * 0.5, 3)
                }
                strokeOpacity={
                  isDimmed ? 0.06 : isActive ? 0.9 : 0.35
                }
                markerEnd={
                  isActive ? "url(#arrow-hl)" : "url(#arrow)"
                }
                className="transition-all duration-200"
              />
            );
          })}
        </svg>

        {/* Container backgrounds */}
        {containers.map((c) => {
          const isDimmed =
            connected !== null && !connected.has(c.id);
          return (
            <div
              key={`bg-${c.id}`}
              className={cn(
                "absolute rounded-xl border border-dashed transition-all duration-200",
                isDimmed
                  ? "opacity-20 border-border/30"
                  : "border-border/50",
              )}
              style={{
                left: c.x - c.w / 2,
                top: c.y - c.h / 2,
                width: c.w,
                height: c.h,
                backgroundColor: "rgba(22, 27, 34, 0.3)",
                zIndex: 1,
              }}
            >
              <div className="flex items-center gap-2 px-3 py-1 text-xs">
                <button
                  className="text-muted hover:text-foreground"
                  onClick={() => toggleExpand(c.id)}
                >
                  <ChevronDown className="w-3.5 h-3.5" />
                </button>
                <span className="text-muted font-medium">
                  {c.domain.label}
                </span>
                {c.domain.health && (
                  <span
                    className={cn(
                      "text-[10px] font-mono",
                      healthColor(
                        c.domain.health.topo_health_score,
                      ),
                    )}
                  >
                    {Math.round(
                      c.domain.health.topo_health_score * 100,
                    )}
                  </span>
                )}
              </div>
            </div>
          );
        })}

        {/* Cards (standalone + children inside containers) */}
        {cards.map((n) => {
          const isDimmed =
            connected !== null && !connected.has(n.id);
          return (
            <div
              key={n.id}
              data-card
              className="absolute transition-all duration-300"
              style={{
                left: n.x - n.w / 2,
                top: n.y - n.h / 2,
                width: n.w,
                zIndex: 2,
              }}
            >
              <DomainCard
                domain={n.domain}
                issueCount={issueCounts.get(n.id) ?? 0}
                isExpanded={expanded.has(n.id)}
                isDimmed={isDimmed}
                isSelected={selectedPath === n.id}
                hasChildren={n.domain.children.length > 0}
                onToggle={() => toggleExpand(n.id)}
                onSelect={() => onSelect(n.domain)}
                onHover={setHoveredId}
              />
            </div>
          );
        })}
        </div>{/* /Transform layer */}
      </div>
    </div>
  );
}
