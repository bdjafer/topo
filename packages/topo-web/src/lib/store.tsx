import { createContext, useContext, useState, type ReactNode } from "react";
import type { TopoResult } from "./types";

interface TopoStore {
  result: TopoResult | null;
  setResult: (r: TopoResult) => void;
}

const TopoContext = createContext<TopoStore | null>(null);

export function TopoProvider({ children }: { children: ReactNode }) {
  const [result, setResult] = useState<TopoResult | null>(null);
  return (
    <TopoContext.Provider value={{ result, setResult }}>
      {children}
    </TopoContext.Provider>
  );
}

export function useTopo() {
  const ctx = useContext(TopoContext);
  if (!ctx) throw new Error("useTopo must be used within TopoProvider");
  return ctx;
}
