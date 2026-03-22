import type { DomainNode, Issue, HealthScore } from "../../lib/types";
import { cn } from "../../lib/utils";
import { HealthGauge } from "./health-gauge";
import {
  Shield,
  AlertTriangle,
  ArrowRight,
  Tag,
  MapPin,
  Layers,
} from "lucide-react";

interface DetailPanelProps {
  domain: DomainNode | null;
  issues: Issue[];
  globalHealth: HealthScore;
}

export function DetailPanel({
  domain,
  issues,
  globalHealth,
}: DetailPanelProps) {
  if (!domain) {
    return (
      <div className="flex items-center justify-center h-full text-muted text-sm p-6">
        <div className="text-center">
          <MapPin className="w-8 h-8 mx-auto mb-3 opacity-30" />
          <p>Select a domain to see details</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <h2 className="text-lg font-semibold">{domain.label}</h2>
          {domain.archetype && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple/10 text-purple border border-purple/20">
              {domain.archetype.label}
            </span>
          )}
        </div>
        <p className="text-xs text-muted font-mono">{domain.path}</p>
        <div className="flex items-center gap-3 mt-2 text-xs text-muted">
          <span>{domain.size} nodes</span>
          <span>
            {domain.children.length > 0
              ? `${domain.children.length} sub-domains`
              : "leaf"}
          </span>
          {domain.depth > 0 && <span>depth {domain.depth}</span>}
        </div>
      </div>

      {/* Health */}
      {domain.health && (
        <Section title="Health" icon={<Shield className="w-3.5 h-3.5" />}>
          <HealthGauge
            label="Overall"
            value={domain.health.topo_health_score}
            className="mb-3"
          />
          <div className="grid grid-cols-2 gap-3">
            <HealthGauge
              label="Coherence"
              value={domain.health.coherence}
              compact
            />
            <HealthGauge label="Flow" value={domain.health.flow} compact />
          </div>
        </Section>
      )}

      {/* Top terms */}
      {domain.top_terms.length > 0 && (
        <Section title="Top Terms" icon={<Tag className="w-3.5 h-3.5" />}>
          <div className="flex flex-wrap gap-1.5">
            {domain.top_terms.map((term) => (
              <span
                key={term}
                className="px-2 py-0.5 text-xs bg-surface rounded-full border border-border text-muted"
              >
                {term}
              </span>
            ))}
          </div>
        </Section>
      )}

      {/* Issues */}
      {issues.length > 0 && (
        <Section
          title={`Issues (${issues.length})`}
          icon={<AlertTriangle className="w-3.5 h-3.5 text-warning" />}
        >
          <div className="space-y-2">
            {issues.map((issue) => (
              <div
                key={issue.id}
                className="p-2.5 rounded-md bg-canvas border border-border"
              >
                <div className="flex items-start gap-2 mb-1">
                  <span
                    className={cn(
                      "text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded shrink-0",
                      issue.severity_label === "high"
                        ? "bg-danger/10 text-danger"
                        : issue.severity_label === "medium"
                          ? "bg-warning/10 text-warning"
                          : "bg-muted/10 text-muted",
                    )}
                  >
                    {issue.severity_label}
                  </span>
                  <span className="text-[10px] text-muted font-mono">
                    {issue.kind}
                  </span>
                </div>
                <p className="text-xs font-medium mb-1">{issue.title}</p>
                <p className="text-[11px] text-muted leading-relaxed">
                  {issue.description}
                </p>
                {issue.anchors.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {issue.anchors.map((a, i) => (
                      <span
                        key={i}
                        className="text-[10px] font-mono text-accent/70"
                      >
                        {a.node_id}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Dependencies */}
      {domain.dependencies.length > 0 && (
        <Section
          title="Dependencies"
          icon={<Layers className="w-3.5 h-3.5" />}
        >
          <div className="space-y-1.5">
            {domain.dependencies.map((dep, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="font-mono text-muted truncate">
                  {dep.source_path}
                </span>
                <ArrowRight className="w-3 h-3 text-muted shrink-0" />
                <span className="font-mono text-accent/70 truncate">
                  {dep.target_path}
                </span>
                <span className="text-[10px] text-muted ml-auto shrink-0">
                  {dep.weight}
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Boundary nodes */}
      {domain.boundary_nodes.length > 0 && (
        <Section
          title="Boundary Nodes"
          icon={<MapPin className="w-3.5 h-3.5" />}
        >
          <p className="text-[11px] text-muted mb-2">
            Nodes near domain boundaries that may shift under refactoring.
          </p>
          <div className="flex flex-wrap gap-1">
            {domain.boundary_nodes.map((n) => (
              <span
                key={n}
                className="px-2 py-0.5 text-[11px] font-mono bg-warning/5 text-warning/80 rounded border border-warning/20"
              >
                {n}
              </span>
            ))}
          </div>
        </Section>
      )}

      {/* Members (leaf) */}
      {domain.members.length > 0 && (
        <Section
          title={`Members (${domain.members.length})`}
          icon={<Layers className="w-3.5 h-3.5" />}
        >
          <div className="space-y-0.5">
            {domain.members.map((m) => (
              <div key={m} className="text-[11px] font-mono text-muted py-0.5">
                {m}
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2 text-xs font-semibold text-foreground uppercase tracking-wider">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}
