import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "@tanstack/react-router";
import { router } from "./router";
import { TopoProvider } from "./lib/store";
import "./globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <TopoProvider>
      <RouterProvider router={router} />
    </TopoProvider>
  </StrictMode>,
);
