import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Interpolate health score [0,1] to a translucent CSS background color. */
export function healthBg(score: number | null | undefined): string {
  if (score == null) return "rgba(48, 54, 61, 0.3)";

  let r: number, g: number, b: number;
  if (score >= 0.5) {
    const t = (score - 0.5) * 2; // 0→1 as score goes 0.5→1
    r = Math.round(210 + (63 - 210) * t);
    g = Math.round(153 + (185 - 153) * t);
    b = Math.round(34 + (80 - 34) * t);
  } else {
    const t = score * 2; // 0→1 as score goes 0→0.5
    r = Math.round(248 + (210 - 248) * t);
    g = Math.round(81 + (153 - 81) * t);
    b = Math.round(73 + (34 - 73) * t);
  }
  return `rgba(${r}, ${g}, ${b}, 0.15)`;
}

/** Health score to Tailwind text color class. */
export function healthColor(score: number | null | undefined): string {
  if (score == null) return "text-muted";
  if (score >= 0.8) return "text-success";
  if (score >= 0.5) return "text-warning";
  return "text-danger";
}

/** Severity label to Tailwind text color class. */
export function severityColor(label: string): string {
  switch (label) {
    case "high":
      return "text-danger";
    case "medium":
      return "text-warning";
    default:
      return "text-muted";
  }
}

/** Consistent color palette for top-level domains. */
export const DOMAIN_COLORS: Record<string, string> = {
  auth: "#58a6ff",
  billing: "#3fb950",
  orders: "#d29922",
  api: "#bc8cff",
  data: "#f85149",
  "cross-cutting": "#8b949e",
};

/** Map a domain path (e.g. "billing/payment") to its top-level domain color. */
export function domainColor(path: string): string {
  return DOMAIN_COLORS[path.split("/")[0]] ?? "#8b949e";
}
