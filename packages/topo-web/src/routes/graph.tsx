import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { useTopo } from "../lib/store";
import { ForceGraph } from "../components/graph/force-graph";

export function GraphPage() {
  const { result } = useTopo();
  const navigate = useNavigate();

  useEffect(() => {
    if (!result) navigate({ to: "/" });
  }, [result, navigate]);

  if (!result) return null;

  return (
    <div className="h-full">
      <ForceGraph nodes={result.graph.nodes} edges={result.graph.edges} />
    </div>
  );
}
