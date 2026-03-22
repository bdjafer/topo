import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import type { GraphNode, GraphEdge } from "../../lib/types";
import { domainColor, DOMAIN_COLORS } from "../../lib/utils";

interface ForceGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  role: string;
  domain_path: string;
}

const EDGE_STYLES: Record<string, { stroke: string; dash: string }> = {
  calls: { stroke: "rgba(139, 148, 158, 0.3)", dash: "" },
  imports: { stroke: "rgba(139, 148, 158, 0.2)", dash: "4,3" },
  inherits: { stroke: "rgba(188, 140, 255, 0.2)", dash: "2,2" },
  defines: { stroke: "rgba(63, 185, 80, 0.15)", dash: "6,3" },
};

function nodeRadius(role: string) {
  switch (role) {
    case "hub":
      return 8;
    case "bridge":
      return 7;
    case "entry_point":
      return 6;
    case "utility":
      return 5;
    default:
      return 4;
  }
}

export function ForceGraph({ nodes, edges }: ForceGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [edgeFilter, setEdgeFilter] = useState<Set<string>>(
    new Set(["calls", "imports"]),
  );

  useEffect(() => {
    const svg = svgRef.current;
    const container = containerRef.current;
    if (!svg || !container) return;

    const { width, height } = container.getBoundingClientRect();
    if (!width || !height) return;

    // Prepare simulation data.
    const simNodes: SimNode[] = nodes.map((n) => ({
      id: n.id,
      role: n.role,
      domain_path: n.domain_path,
    }));
    const nodeMap = new Map(simNodes.map((n) => [n.id, n]));

    const simLinks = edges
      .filter((e) => edgeFilter.has(e.kind))
      .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        kind: e.kind,
      }));

    // Clear previous render.
    const sel = d3.select(svg);
    sel.selectAll("*").remove();
    sel.attr("width", width).attr("height", height);

    const g = sel.append("g");

    // Zoom.
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    sel.call(zoom);

    // Arrow marker.
    sel
      .append("defs")
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 0 10 6")
      .attr("refX", 18)
      .attr("refY", 3)
      .attr("markerWidth", 8)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,0 L10,3 L0,6")
      .attr("fill", "rgba(139, 148, 158, 0.4)");

    // Edges.
    const link = g
      .append("g")
      .selectAll("line")
      .data(simLinks)
      .join("line")
      .attr(
        "stroke",
        (d) => EDGE_STYLES[d.kind]?.stroke ?? "rgba(139,148,158,0.2)",
      )
      .attr("stroke-dasharray", (d) => EDGE_STYLES[d.kind]?.dash ?? "")
      .attr("stroke-width", 1)
      .attr("marker-end", "url(#arrowhead)");

    // Nodes.
    const node = g
      .append("g")
      .selectAll<SVGGElement, SimNode>("g")
      .data(simNodes)
      .join("g")
      .call(
        d3
          .drag<SVGGElement, SimNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }),
      );

    node
      .append("circle")
      .attr("r", (d) => nodeRadius(d.role))
      .attr("fill", (d) => domainColor(d.domain_path))
      .attr("stroke", (d) => domainColor(d.domain_path))
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 0.5)
      .attr("fill-opacity", 0.8);

    node
      .append("title")
      .text((d) => `${d.id}\n${d.role} · ${d.domain_path}`);

    // Labels for notable roles.
    node
      .filter((d) => d.role !== "regular")
      .append("text")
      .attr("dx", 10)
      .attr("dy", 3)
      .attr("font-size", "9px")
      .attr("fill", "rgba(230, 237, 243, 0.6)")
      .attr("font-family", "var(--font-mono)")
      .text((d) => d.id.split(".").pop()!);

    // Simulation.
    const simulation = d3
      .forceSimulation(simNodes)
      .force(
        "link",
        d3
          .forceLink(simLinks)
          .id((d: any) => d.id)
          .distance(50),
      )
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(12))
      .on("tick", () => {
        link
          .attr("x1", (d: any) => d.source.x)
          .attr("y1", (d: any) => d.source.y)
          .attr("x2", (d: any) => d.target.x)
          .attr("y2", (d: any) => d.target.y);

        node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });

    return () => {
      simulation.stop();
    };
  }, [nodes, edges, edgeFilter]);

  const edgeKinds = ["calls", "imports", "inherits", "defines"];

  return (
    <div className="h-full flex flex-col">
      {/* Controls */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-border bg-surface/50 shrink-0">
        <span className="text-xs text-muted font-medium">Edges:</span>
        {edgeKinds.map((kind) => (
          <label
            key={kind}
            className="flex items-center gap-1.5 text-xs cursor-pointer"
          >
            <input
              type="checkbox"
              checked={edgeFilter.has(kind)}
              onChange={() => {
                const next = new Set(edgeFilter);
                if (next.has(kind)) next.delete(kind);
                else next.add(kind);
                setEdgeFilter(next);
              }}
              className="rounded border-border"
            />
            <span className="text-muted">{kind}</span>
          </label>
        ))}

        {/* Legend */}
        <div className="ml-auto flex items-center gap-3">
          {Object.entries(DOMAIN_COLORS)
            .filter(([k]) => k !== "cross-cutting")
            .map(([name, color]) => (
              <div key={name} className="flex items-center gap-1">
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="text-[10px] text-muted">{name}</span>
              </div>
            ))}
        </div>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="flex-1">
        <svg ref={svgRef} className="w-full h-full" />
      </div>
    </div>
  );
}
