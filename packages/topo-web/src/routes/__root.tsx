import { Outlet, Link, useRouterState } from "@tanstack/react-router";
import { cn } from "../lib/utils";
import { Hexagon, Network, LayoutGrid } from "lucide-react";

export function RootLayout() {
  const { location } = useRouterState();
  const path = location.pathname;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-5 py-3 shrink-0">
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="flex items-center gap-2 text-foreground hover:text-accent transition-colors"
          >
            <Hexagon className="w-5 h-5 text-accent" />
            <span className="text-lg font-semibold">topo</span>
          </Link>
          <span className="text-xs text-muted bg-surface px-2 py-0.5 rounded-full border border-border">
            structural intelligence
          </span>
        </div>

        <nav className="flex items-center gap-1">
          <NavLink to="/domain" active={path === "/domain"}>
            <LayoutGrid className="w-3.5 h-3.5" />
            Domains
          </NavLink>
          <NavLink to="/graph" active={path === "/graph"}>
            <Network className="w-3.5 h-3.5" />
            Graph
          </NavLink>
        </nav>
      </header>

      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}

function NavLink({
  to,
  active,
  children,
}: {
  to: string;
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <Link
      to={to}
      className={cn(
        "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
        active
          ? "bg-accent-dim/30 text-accent"
          : "text-muted hover:text-foreground hover:bg-surface",
      )}
    >
      {children}
    </Link>
  );
}
