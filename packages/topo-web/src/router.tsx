import {
  createRouter,
  createRoute,
  createRootRoute,
} from "@tanstack/react-router";
import { RootLayout } from "./routes/__root";
import { LandingPage } from "./routes/index";
import { DomainPage } from "./routes/domain";
import { GraphPage } from "./routes/graph";

const rootRoute = createRootRoute({
  component: RootLayout,
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: LandingPage,
});

const domainRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/domain",
  component: DomainPage,
});

const graphRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/graph",
  component: GraphPage,
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  domainRoute,
  graphRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
