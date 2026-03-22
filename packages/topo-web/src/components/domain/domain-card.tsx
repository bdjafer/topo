import type { DomainNode } from "../../lib/types";
import { cn, healthColor } from "../../lib/utils";
import { AlertTriangle, ChevronDown, ChevronRight } from "lucide-react";

interface Props {
  domain: DomainNode;
  issueCount: number;
  isExpanded: boolean;
  isDimmed: boolean;
  isSelected: boolean;
  hasChildren: boolean;
  onToggle: () => void;
  onSelect: () => void;
  onHover: (id: string | null) => void;
}

function barColor(score: number | null | undefined): string {
  if (score == null) return "#30363d";
  if (score >= 0.8) return "#3fb950";
  if (score >= 0.5) return "#d29922";
  return "#f85149";
}

export function DomainCard({
  domain,
  issueCount,
  isExpanded,
  isDimmed,
  isSelected,
  hasChildren,
  onToggle,
  onSelect,
  onHover,
}: Props) {
  const score = domain.health?.topo_health_score;
  const color = barColor(score);

  return (
    <div
      className={cn(
        "rounded-lg bg-surface border border-border select-none cursor-pointer transition-all duration-200",
        isDimmed && "opacity-25",
        isSelected && "ring-2 ring-accent",
        !isDimmed && "hover:border-accent/40",
      )}
      style={{ borderLeftWidth: 3, borderLeftColor: color }}
      onMouseEnter={() => onHover(domain.path)}
      onMouseLeave={() => onHover(null)}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
      onDoubleClick={(e) => {
        e.stopPropagation();
        if (hasChildren) onToggle();
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-1.5 px-3 py-1.5">
        {hasChildren && (
          <button
            className="text-muted hover:text-foreground shrink-0"
            onClick={(e) => {
              e.stopPropagation();
              onToggle();
            }}
          >
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
          </button>
        )}
        <span className="text-sm font-semibold truncate">{domain.label}</span>
        {issueCount > 0 && (
          <span className="flex items-center gap-0.5 text-danger ml-auto shrink-0">
            <AlertTriangle className="w-3 h-3" />
            <span className="text-[10px] font-mono">{issueCount}</span>
          </span>
        )}
      </div>

      {/* Health bar */}
      {score != null && (
        <div className="px-3 flex items-center gap-2">
          <div className="flex-1 h-1.5 rounded-full bg-canvas overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.round(score * 100)}%`,
                backgroundColor: color,
              }}
            />
          </div>
          <span
            className={cn(
              "text-[10px] font-mono tabular-nums",
              healthColor(score),
            )}
          >
            {Math.round(score * 100)}
          </span>
        </div>
      )}

      {/* Meta */}
      <div className="px-3 pt-0.5 pb-1.5 text-[10px] text-muted flex gap-2">
        <span>{domain.size} nodes</span>
        {hasChildren && <span>{domain.children.length} sub</span>}
      </div>
    </div>
  );
}
