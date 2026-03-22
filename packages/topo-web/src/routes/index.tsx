import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { useTopo } from "../lib/store";
import { createMockResult } from "../lib/mock";
import { Hexagon, ArrowRight, Github } from "lucide-react";

export function LandingPage() {
  const [repoUrl, setRepoUrl] = useState("");
  const { setResult } = useTopo();
  const navigate = useNavigate();

  function handleDemo() {
    setResult(createMockResult());
    navigate({ to: "/domain" });
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    // TODO: call topo serve API with the repo URL
    setResult(createMockResult());
    navigate({ to: "/domain" });
  }

  return (
    <div className="flex items-center justify-center h-full">
      <div className="max-w-lg w-full mx-auto px-6">
        {/* Hero */}
        <div className="text-center mb-10">
          <Hexagon
            className="w-16 h-16 text-accent mx-auto mb-4"
            strokeWidth={1.5}
          />
          <h1 className="text-3xl font-bold mb-2">topo</h1>
          <p className="text-muted text-lg">
            Structural intelligence for your codebase
          </p>
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="mb-6">
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Github className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
              <input
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="owner/repo or https://github.com/..."
                className="w-full bg-surface border border-border rounded-lg pl-10 pr-4 py-2.5 text-sm text-foreground placeholder:text-muted/50 focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/30"
              />
            </div>
            <button
              type="submit"
              className="bg-accent-dim hover:bg-accent text-white px-4 py-2.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5"
            >
              Analyze
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </form>

        {/* Demo */}
        <div className="text-center">
          <button
            onClick={handleDemo}
            className="text-accent hover:text-accent/80 text-sm underline underline-offset-2 transition-colors"
          >
            Try with demo data
          </button>
          <p className="text-muted text-xs mt-2">
            Explore a sample web app with 50 nodes across 5 domains
          </p>
        </div>
      </div>
    </div>
  );
}
