import { useNavigate } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { useTopo } from "../lib/store";
import { DomainGraph } from "../components/domain/domain-graph";
import { DetailPanel } from "../components/domain/detail-panel";
import type { DomainNode } from "../lib/types";

export function DomainPage() {
  const { result } = useTopo();
  const navigate = useNavigate();
  const [selected, setSelected] = useState<DomainNode | null>(null);

  useEffect(() => {
    if (!result) navigate({ to: "/" });
  }, [result, navigate]);

  if (!result) return null;

  // Find issues whose anchors fall within the selected domain.
  const selectedIssues = selected
    ? result.issues.filter((issue) =>
        issue.anchors.some((a) => {
          const node = result.graph.nodes.find((n) => n.id === a.node_id);
          return node && node.domain_path.startsWith(selected.path);
        }),
      )
    : [];

  return (
    <div className="flex h-full">
      <div className="flex-1 min-w-0">
        <DomainGraph
          domain={result.domain}
          issues={result.issues}
          graphNodes={result.graph.nodes}
          onSelect={setSelected}
          selectedPath={selected?.path ?? null}
        />
      </div>

      <div className="w-80 border-l border-border overflow-y-auto shrink-0">
        <DetailPanel
          domain={selected}
          issues={selectedIssues}
          globalHealth={result.health}
        />
      </div>
    </div>
  );
}
