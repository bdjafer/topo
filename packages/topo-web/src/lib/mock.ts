import type {
  TopoResult,
  GraphNode,
  GraphEdge,
  DomainNode,
  Issue,
  StructuralRole,
} from "./types";

// ── Domain definitions ────────────────────────────────────────────

const DOMAINS: Record<string, { nodes: string[]; file: string }> = {
  "auth/token": {
    nodes: [
      "auth.jwt_validate",
      "auth.jwt_sign",
      "auth.refresh_token",
      "auth.token_store",
      "auth.token_blacklist",
    ],
    file: "src/auth/token.rs",
  },
  "auth/session": {
    nodes: [
      "auth.session_create",
      "auth.session_destroy",
      "auth.session_check",
      "auth.cookie_parse",
      "auth.csrf_check",
    ],
    file: "src/auth/session.rs",
  },
  "billing/invoicing": {
    nodes: [
      "billing.create_invoice",
      "billing.send_invoice",
      "billing.invoice_pdf",
      "billing.tax_calc",
    ],
    file: "src/billing/invoice.rs",
  },
  "billing/payment": {
    nodes: [
      "billing.process_charge",
      "billing.refund",
      "billing.payment_gateway",
      "billing.payment_webhook",
      "billing.retry_charge",
    ],
    file: "src/billing/payment.rs",
  },
  "orders/management": {
    nodes: [
      "orders.create_order",
      "orders.update_order",
      "orders.cancel_order",
      "orders.order_query",
    ],
    file: "src/orders/manage.rs",
  },
  "orders/fulfillment": {
    nodes: [
      "orders.ship_order",
      "orders.track_shipment",
      "orders.warehouse_api",
      "orders.label_print",
      "orders.delivery_notify",
      "orders.return_process",
    ],
    file: "src/orders/fulfill.rs",
  },
  "api/routing": {
    nodes: [
      "api.router",
      "api.route_auth",
      "api.route_billing",
      "api.route_orders",
      "api.health_check",
    ],
    file: "src/api/routes.rs",
  },
  "api/middleware": {
    nodes: [
      "api.cors",
      "api.rate_limiter",
      "api.error_handler",
      "api.auth_middleware",
    ],
    file: "src/api/middleware.rs",
  },
  "data/database": {
    nodes: [
      "data.connect",
      "data.query",
      "data.transaction",
      "data.migrate",
      "data.pool",
    ],
    file: "src/data/db.rs",
  },
  "data/cache": {
    nodes: ["data.cache_get", "data.cache_set", "data.cache_invalidate"],
    file: "src/data/cache.rs",
  },
};

const CROSS_CUTTING = [
  "utils.logger",
  "utils.format_response",
  "utils.config_loader",
];

const ROLE_OVERRIDES: Record<string, StructuralRole> = {
  "api.router": "hub",
  "api.auth_middleware": "bridge",
  "data.query": "utility",
  "data.connect": "utility",
  "data.cache_get": "utility",
  "api.route_auth": "entry_point",
  "api.route_billing": "entry_point",
  "api.route_orders": "entry_point",
  "api.health_check": "entry_point",
  "utils.logger": "utility",
  "utils.format_response": "utility",
  "utils.config_loader": "utility",
};

// ── Graph builders ────────────────────────────────────────────────

function makeNodes(): GraphNode[] {
  const nodes: GraphNode[] = [];
  let clusterId = 0;

  for (const [domainPath, def] of Object.entries(DOMAINS)) {
    for (let i = 0; i < def.nodes.length; i++) {
      const id = def.nodes[i];
      nodes.push({
        id,
        kind: "function",
        file: def.file,
        line: 10 + i * 15,
        role: ROLE_OVERRIDES[id] ?? "regular",
        domain_path: domainPath,
        cluster_id: clusterId,
      });
    }
    clusterId++;
  }

  for (const id of CROSS_CUTTING) {
    nodes.push({
      id,
      kind: "function",
      file: "src/utils/mod.rs",
      line: 1,
      role: ROLE_OVERRIDES[id] ?? "utility",
      domain_path: "cross-cutting",
      cluster_id: -1,
    });
  }

  return nodes;
}

function makeEdges(): GraphEdge[] {
  return [
    // ── Auth internal ──
    { source: "auth.jwt_validate", target: "auth.token_store", kind: "calls" },
    { source: "auth.jwt_sign", target: "auth.token_store", kind: "calls" },
    {
      source: "auth.refresh_token",
      target: "auth.jwt_validate",
      kind: "calls",
    },
    { source: "auth.refresh_token", target: "auth.jwt_sign", kind: "calls" },
    {
      source: "auth.token_blacklist",
      target: "auth.token_store",
      kind: "calls",
    },
    { source: "auth.session_create", target: "auth.jwt_sign", kind: "calls" },
    {
      source: "auth.session_check",
      target: "auth.jwt_validate",
      kind: "calls",
    },
    {
      source: "auth.session_destroy",
      target: "auth.token_blacklist",
      kind: "calls",
    },
    {
      source: "auth.cookie_parse",
      target: "auth.session_check",
      kind: "calls",
    },
    { source: "auth.csrf_check", target: "auth.session_check", kind: "calls" },

    // ── Billing internal ──
    {
      source: "billing.create_invoice",
      target: "billing.tax_calc",
      kind: "calls",
    },
    {
      source: "billing.send_invoice",
      target: "billing.invoice_pdf",
      kind: "calls",
    },
    {
      source: "billing.process_charge",
      target: "billing.payment_gateway",
      kind: "calls",
    },
    {
      source: "billing.refund",
      target: "billing.payment_gateway",
      kind: "calls",
    },
    {
      source: "billing.retry_charge",
      target: "billing.process_charge",
      kind: "calls",
    },
    {
      source: "billing.payment_webhook",
      target: "billing.process_charge",
      kind: "calls",
    },
    // Cycle: charge ↔ webhook
    {
      source: "billing.process_charge",
      target: "billing.payment_webhook",
      kind: "calls",
    },

    // ── Orders internal ──
    {
      source: "orders.create_order",
      target: "orders.order_query",
      kind: "calls",
    },
    {
      source: "orders.update_order",
      target: "orders.order_query",
      kind: "calls",
    },
    {
      source: "orders.cancel_order",
      target: "orders.order_query",
      kind: "calls",
    },
    {
      source: "orders.ship_order",
      target: "orders.track_shipment",
      kind: "calls",
    },
    {
      source: "orders.ship_order",
      target: "orders.warehouse_api",
      kind: "calls",
    },
    {
      source: "orders.ship_order",
      target: "orders.label_print",
      kind: "calls",
    },
    {
      source: "orders.delivery_notify",
      target: "orders.track_shipment",
      kind: "calls",
    },
    {
      source: "orders.return_process",
      target: "orders.warehouse_api",
      kind: "calls",
    },
    // Cycle in fulfillment
    {
      source: "orders.track_shipment",
      target: "orders.ship_order",
      kind: "calls",
    },

    // ── API routing ──
    { source: "api.router", target: "api.route_auth", kind: "calls" },
    { source: "api.router", target: "api.route_billing", kind: "calls" },
    { source: "api.router", target: "api.route_orders", kind: "calls" },
    {
      source: "api.route_auth",
      target: "auth.session_create",
      kind: "calls",
    },
    {
      source: "api.route_auth",
      target: "auth.session_destroy",
      kind: "calls",
    },
    {
      source: "api.route_billing",
      target: "billing.create_invoice",
      kind: "calls",
    },
    {
      source: "api.route_billing",
      target: "billing.process_charge",
      kind: "calls",
    },
    {
      source: "api.route_orders",
      target: "orders.create_order",
      kind: "calls",
    },
    {
      source: "api.route_orders",
      target: "orders.ship_order",
      kind: "calls",
    },
    {
      source: "api.auth_middleware",
      target: "auth.session_check",
      kind: "calls",
    },
    { source: "api.cors", target: "utils.config_loader", kind: "calls" },
    { source: "api.rate_limiter", target: "data.cache_get", kind: "calls" },
    {
      source: "api.error_handler",
      target: "utils.format_response",
      kind: "calls",
    },

    // ── Data internal ──
    { source: "data.query", target: "data.connect", kind: "calls" },
    { source: "data.transaction", target: "data.query", kind: "calls" },
    { source: "data.migrate", target: "data.connect", kind: "calls" },
    { source: "data.pool", target: "data.connect", kind: "calls" },
    { source: "data.cache_set", target: "data.cache_get", kind: "calls" },
    {
      source: "data.cache_invalidate",
      target: "data.cache_get",
      kind: "calls",
    },

    // ── Cross-domain ──
    { source: "auth.token_store", target: "data.query", kind: "calls" },
    {
      source: "billing.create_invoice",
      target: "data.transaction",
      kind: "calls",
    },
    {
      source: "billing.process_charge",
      target: "data.transaction",
      kind: "calls",
    },
    {
      source: "orders.create_order",
      target: "data.transaction",
      kind: "calls",
    },
    { source: "orders.order_query", target: "data.query", kind: "calls" },

    // Layer violation: fulfillment → billing directly
    {
      source: "orders.ship_order",
      target: "billing.process_charge",
      kind: "calls",
    },
    // Misplaced concern: webhook does auth
    {
      source: "billing.payment_webhook",
      target: "auth.jwt_validate",
      kind: "calls",
    },

    // ── Cross-cutting usage ──
    { source: "auth.session_create", target: "utils.logger", kind: "calls" },
    {
      source: "billing.process_charge",
      target: "utils.logger",
      kind: "calls",
    },
    { source: "orders.ship_order", target: "utils.logger", kind: "calls" },
    { source: "api.router", target: "utils.logger", kind: "calls" },
    { source: "data.query", target: "utils.logger", kind: "calls" },
    { source: "api.error_handler", target: "utils.logger", kind: "calls" },

    // ── Import edges ──
    {
      source: "api.route_auth",
      target: "auth.session_create",
      kind: "imports",
    },
    {
      source: "api.route_billing",
      target: "billing.create_invoice",
      kind: "imports",
    },
    {
      source: "api.route_orders",
      target: "orders.create_order",
      kind: "imports",
    },
    { source: "auth.session_create", target: "auth.jwt_sign", kind: "imports" },
    {
      source: "billing.create_invoice",
      target: "billing.tax_calc",
      kind: "imports",
    },
  ];
}

// ── Domain tree ───────────────────────────────────────────────────

function leaf(
  label: string,
  path: string,
  domainKey: string,
  health: { topo_health_score: number; coherence: number; flow: number },
  opts: {
    terms?: string[];
    boundary?: string[];
    deps?: DomainNode["dependencies"];
  } = {},
): DomainNode {
  return {
    label,
    path,
    depth: 2,
    size: DOMAINS[domainKey].nodes.length,
    members: [...DOMAINS[domainKey].nodes],
    top_terms: opts.terms ?? [],
    coherence: health.coherence,
    archetype: null,
    health,
    children: [],
    cross_cutting: [],
    boundary_nodes: opts.boundary ?? [],
    dependencies: opts.deps ?? [],
  };
}

function makeDomainTree(): DomainNode {
  return {
    label: "system",
    path: "system",
    depth: 0,
    size: 50,
    members: [],
    top_terms: [],
    coherence: null,
    archetype: null,
    health: { topo_health_score: 0.68, coherence: 0.72, flow: 0.64 },
    children: [
      {
        label: "auth",
        path: "auth",
        depth: 1,
        size: 10,
        members: [],
        top_terms: ["authenticate", "token", "session"],
        coherence: 0.85,
        archetype: { label: "authentication/authorization", confidence: 0.89 },
        health: { topo_health_score: 0.82, coherence: 0.85, flow: 0.79 },
        children: [
          leaf(
            "token",
            "auth/token",
            "auth/token",
            { topo_health_score: 0.91, coherence: 0.91, flow: 0.92 },
            {
              terms: ["jwt", "validate", "refresh", "sign"],
              deps: [
                {
                  source_path: "auth/token",
                  target_path: "data/database",
                  weight: 2,
                  edge_kinds: { calls: 2 },
                },
              ],
            },
          ),
          leaf(
            "session",
            "auth/session",
            "auth/session",
            { topo_health_score: 0.88, coherence: 0.88, flow: 0.88 },
            {
              terms: ["session", "cookie", "csrf"],
              deps: [
                {
                  source_path: "auth/session",
                  target_path: "auth/token",
                  weight: 4,
                  edge_kinds: { calls: 3, imports: 1 },
                },
              ],
            },
          ),
        ],
        cross_cutting: [],
        boundary_nodes: [],
        dependencies: [
          {
            source_path: "auth",
            target_path: "data",
            weight: 2,
            edge_kinds: { calls: 2 },
          },
        ],
      },
      {
        label: "billing",
        path: "billing",
        depth: 1,
        size: 9,
        members: [],
        top_terms: ["payment", "invoice", "charge"],
        coherence: 0.71,
        archetype: { label: "billing/payments", confidence: 0.76 },
        health: { topo_health_score: 0.65, coherence: 0.71, flow: 0.6 },
        children: [
          leaf(
            "invoicing",
            "billing/invoicing",
            "billing/invoicing",
            { topo_health_score: 0.84, coherence: 0.84, flow: 0.84 },
            {
              terms: ["invoice", "pdf", "tax"],
              deps: [
                {
                  source_path: "billing/invoicing",
                  target_path: "data/database",
                  weight: 1,
                  edge_kinds: { calls: 1 },
                },
              ],
            },
          ),
          leaf(
            "payment",
            "billing/payment",
            "billing/payment",
            { topo_health_score: 0.52, coherence: 0.59, flow: 0.45 },
            {
              terms: ["charge", "refund", "gateway", "webhook"],
              boundary: ["billing.payment_webhook"],
              deps: [
                {
                  source_path: "billing/payment",
                  target_path: "billing/invoicing",
                  weight: 3,
                  edge_kinds: { calls: 3 },
                },
                {
                  source_path: "billing/payment",
                  target_path: "data/database",
                  weight: 2,
                  edge_kinds: { calls: 2 },
                },
                {
                  source_path: "billing/payment",
                  target_path: "auth/token",
                  weight: 1,
                  edge_kinds: { calls: 1 },
                },
              ],
            },
          ),
        ],
        cross_cutting: [],
        boundary_nodes: ["billing.payment_webhook"],
        dependencies: [
          {
            source_path: "billing",
            target_path: "data",
            weight: 3,
            edge_kinds: { calls: 3 },
          },
          {
            source_path: "billing",
            target_path: "auth",
            weight: 1,
            edge_kinds: { calls: 1 },
          },
        ],
      },
      {
        label: "orders",
        path: "orders",
        depth: 1,
        size: 10,
        members: [],
        top_terms: ["order", "ship", "fulfill"],
        coherence: 0.54,
        archetype: { label: "order-management", confidence: 0.71 },
        health: { topo_health_score: 0.48, coherence: 0.54, flow: 0.42 },
        children: [
          leaf(
            "management",
            "orders/management",
            "orders/management",
            { topo_health_score: 0.77, coherence: 0.77, flow: 0.77 },
            {
              terms: ["create", "update", "cancel", "query"],
              deps: [
                {
                  source_path: "orders/management",
                  target_path: "data/database",
                  weight: 2,
                  edge_kinds: { calls: 2 },
                },
              ],
            },
          ),
          leaf(
            "fulfillment",
            "orders/fulfillment",
            "orders/fulfillment",
            { topo_health_score: 0.31, coherence: 0.38, flow: 0.25 },
            {
              terms: ["ship", "track", "warehouse", "delivery"],
              boundary: ["orders.return_process"],
              deps: [
                {
                  source_path: "orders/fulfillment",
                  target_path: "orders/management",
                  weight: 2,
                  edge_kinds: { calls: 2 },
                },
                {
                  source_path: "orders/fulfillment",
                  target_path: "billing/payment",
                  weight: 1,
                  edge_kinds: { calls: 1 },
                },
              ],
            },
          ),
        ],
        cross_cutting: [],
        boundary_nodes: [],
        dependencies: [
          {
            source_path: "orders",
            target_path: "data",
            weight: 2,
            edge_kinds: { calls: 2 },
          },
          {
            source_path: "orders",
            target_path: "billing",
            weight: 1,
            edge_kinds: { calls: 1 },
          },
        ],
      },
      {
        label: "api",
        path: "api",
        depth: 1,
        size: 9,
        members: [],
        top_terms: ["route", "middleware", "handler"],
        coherence: 0.78,
        archetype: { label: "api-gateway", confidence: 0.82 },
        health: { topo_health_score: 0.75, coherence: 0.78, flow: 0.72 },
        children: [
          leaf(
            "routing",
            "api/routing",
            "api/routing",
            { topo_health_score: 0.82, coherence: 0.85, flow: 0.79 },
            {
              terms: ["router", "route", "endpoint"],
              deps: [
                {
                  source_path: "api/routing",
                  target_path: "auth/session",
                  weight: 2,
                  edge_kinds: { calls: 2 },
                },
                {
                  source_path: "api/routing",
                  target_path: "billing/invoicing",
                  weight: 1,
                  edge_kinds: { calls: 1 },
                },
                {
                  source_path: "api/routing",
                  target_path: "orders/management",
                  weight: 1,
                  edge_kinds: { calls: 1 },
                },
              ],
            },
          ),
          leaf(
            "middleware",
            "api/middleware",
            "api/middleware",
            { topo_health_score: 0.72, coherence: 0.72, flow: 0.72 },
            {
              terms: ["cors", "rate-limit", "error"],
              deps: [
                {
                  source_path: "api/middleware",
                  target_path: "api/routing",
                  weight: 4,
                  edge_kinds: { calls: 4 },
                },
                {
                  source_path: "api/middleware",
                  target_path: "auth/session",
                  weight: 1,
                  edge_kinds: { calls: 1 },
                },
                {
                  source_path: "api/middleware",
                  target_path: "data/cache",
                  weight: 1,
                  edge_kinds: { calls: 1 },
                },
              ],
            },
          ),
        ],
        cross_cutting: [],
        boundary_nodes: [],
        dependencies: [
          {
            source_path: "api",
            target_path: "auth",
            weight: 3,
            edge_kinds: { calls: 3 },
          },
          {
            source_path: "api",
            target_path: "billing",
            weight: 2,
            edge_kinds: { calls: 2 },
          },
          {
            source_path: "api",
            target_path: "orders",
            weight: 2,
            edge_kinds: { calls: 2 },
          },
        ],
      },
      {
        label: "data",
        path: "data",
        depth: 1,
        size: 8,
        members: [],
        top_terms: ["database", "cache", "query"],
        coherence: 0.8,
        archetype: { label: "data-access", confidence: 0.85 },
        health: { topo_health_score: 0.8, coherence: 0.8, flow: 0.8 },
        children: [
          leaf(
            "database",
            "data/database",
            "data/database",
            { topo_health_score: 0.88, coherence: 0.88, flow: 0.88 },
            { terms: ["query", "transaction", "migrate", "pool"] },
          ),
          leaf(
            "cache",
            "data/cache",
            "data/cache",
            { topo_health_score: 0.73, coherence: 0.73, flow: 0.73 },
            {
              terms: ["cache", "invalidate"],
              deps: [
                {
                  source_path: "data/cache",
                  target_path: "data/database",
                  weight: 3,
                  edge_kinds: { calls: 3 },
                },
              ],
            },
          ),
        ],
        cross_cutting: [],
        boundary_nodes: [],
        dependencies: [],
      },
    ],
    cross_cutting: [
      {
        node_id: "utils.logger",
        silhouette: -0.18,
        caller_diversity: 0.8,
        dominant_edges: { auth: 1, billing: 1, orders: 1, api: 2, data: 1 },
      },
      {
        node_id: "utils.format_response",
        silhouette: -0.12,
        caller_diversity: 0.6,
        dominant_edges: { api: 1 },
      },
      {
        node_id: "utils.config_loader",
        silhouette: -0.08,
        caller_diversity: 0.4,
        dominant_edges: { api: 1 },
      },
    ],
    boundary_nodes: [],
    dependencies: [],
  };
}

// ── Issues ────────────────────────────────────────────────────────

function makeIssues(): Issue[] {
  return [
    {
      id: "circular-dependency:fulfillment-cycle",
      kind: "circular_dependency",
      title: "Dependency cycle in fulfillment",
      description:
        "ship_order and track_shipment form a circular dependency. Ship triggers tracking, but tracking feeds back into shipping logic, creating a tightly coupled cycle that's hard to test independently.",
      severity: 0.85,
      severity_label: "high",
      confidence: 0.95,
      anchors: [
        {
          node_id: "orders.ship_order",
          file: "src/orders/fulfill.rs",
          line: 10,
          kind: "function",
        },
        {
          node_id: "orders.track_shipment",
          file: "src/orders/fulfill.rs",
          line: 25,
          kind: "function",
        },
      ],
    },
    {
      id: "circular-dependency:payment-cycle",
      kind: "circular_dependency",
      title: "Dependency cycle in payment processing",
      description:
        "process_charge and payment_webhook form a circular dependency. Charge processing triggers webhooks, and webhooks trigger charge processing.",
      severity: 0.75,
      severity_label: "high",
      confidence: 0.92,
      anchors: [
        {
          node_id: "billing.process_charge",
          file: "src/billing/payment.rs",
          line: 10,
          kind: "function",
        },
        {
          node_id: "billing.payment_webhook",
          file: "src/billing/payment.rs",
          line: 55,
          kind: "function",
        },
      ],
    },
    {
      id: "layer-violation:fulfillment-billing",
      kind: "layer_violation",
      title: "Layer violation: fulfillment \u2192 billing",
      description:
        "orders.ship_order directly calls billing.process_charge, bypassing the API/service layer. This creates tight coupling between orders and billing domains.",
      severity: 0.7,
      severity_label: "medium",
      confidence: 0.88,
      anchors: [
        {
          node_id: "orders.ship_order",
          file: "src/orders/fulfill.rs",
          line: 10,
          kind: "function",
        },
        {
          node_id: "billing.process_charge",
          file: "src/billing/payment.rs",
          line: 10,
          kind: "function",
        },
      ],
    },
    {
      id: "misplaced-concern:webhook-auth",
      kind: "misplaced_concern",
      title: "Auth logic in billing webhook",
      description:
        "billing.payment_webhook directly calls auth.jwt_validate. Authentication concerns should not leak into billing domain code. Consider using the auth middleware instead.",
      severity: 0.65,
      severity_label: "medium",
      confidence: 0.82,
      anchors: [
        {
          node_id: "billing.payment_webhook",
          file: "src/billing/payment.rs",
          line: 55,
          kind: "function",
        },
        {
          node_id: "auth.jwt_validate",
          file: "src/auth/token.rs",
          line: 10,
          kind: "function",
        },
      ],
    },
    {
      id: "wide-interface:api-router",
      kind: "wide_interface",
      title: "Wide interface: api.router",
      description:
        "api.router has 6 direct dependencies across 4 domains. Consider introducing a service layer to decouple the router from domain implementations.",
      severity: 0.5,
      severity_label: "medium",
      confidence: 0.78,
      anchors: [
        {
          node_id: "api.router",
          file: "src/api/routes.rs",
          line: 10,
          kind: "function",
        },
      ],
    },
    {
      id: "incoherent-module:fulfillment",
      kind: "incoherent_module",
      title: "Low coherence in fulfillment",
      description:
        "The fulfillment sub-domain has coherence 0.38 \u2014 its members don't share a consistent structural profile. Shipping, tracking, and returns may be better separated into distinct concerns.",
      severity: 0.6,
      severity_label: "medium",
      confidence: 0.75,
      anchors: [
        {
          node_id: "orders.ship_order",
          file: "src/orders/fulfill.rs",
          line: 10,
          kind: "function",
        },
        {
          node_id: "orders.return_process",
          file: "src/orders/fulfill.rs",
          line: 85,
          kind: "function",
        },
      ],
    },
  ];
}

// ── Export ─────────────────────────────────────────────────────────

export function createMockResult(): TopoResult {
  const nodes = makeNodes();
  const edges = makeEdges();

  return {
    meta: {
      repo: "acme/webshop",
      analyzed_at: new Date().toISOString(),
      node_count: nodes.length,
      edge_count: edges.length,
    },
    graph: { nodes, edges },
    domain: makeDomainTree(),
    health: { topo_health_score: 0.68, coherence: 0.72, flow: 0.64 },
    issues: makeIssues(),
  };
}
