import { cn, healthColor } from "../../lib/utils";

interface HealthGaugeProps {
  label: string;
  value: number;
  compact?: boolean;
  className?: string;
}

export function HealthGauge({
  label,
  value,
  compact,
  className,
}: HealthGaugeProps) {
  const pct = Math.round(value * 100);
  const color = healthColor(value);
  const barColor =
    value >= 0.8 ? "bg-success" : value >= 0.5 ? "bg-warning" : "bg-danger";

  if (compact) {
    return (
      <div className={cn("", className)}>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[11px] text-muted">{label}</span>
          <span className={cn("text-xs font-mono font-semibold tabular-nums", color)}>
            {pct}
          </span>
        </div>
        <div className="h-1.5 rounded-full bg-surface overflow-hidden">
          <div
            className={cn("h-full rounded-full transition-all duration-500", barColor)}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-medium text-foreground">{label}</span>
          <span className={cn("text-sm font-mono font-bold tabular-nums", color)}>
            {pct}
          </span>
        </div>
        <div className="h-2 rounded-full bg-surface overflow-hidden">
          <div
            className={cn("h-full rounded-full transition-all duration-500", barColor)}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
