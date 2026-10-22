import {
  HealthResponse,
  SystemStatusResponse,
  TelemetrySource,
  IncidentEvent,
  Decision,
  SimulationRun,
  EvaluationResponse,
  ResearchExperimentsResponse,
  ActionSpaceItem,
  ScenarioItem,
  StartSimulationPayload,
  StartSimulationResponse,
  SimulationResults,
  SimulationComparisonResponse,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
      next: { revalidate: 0 },
    });
    if (!res.ok) {
      throw new Error(`API error ${res.status}: ${res.statusText}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    // If external call fails, attempt relative path
    if (typeof window !== "undefined") {
      try {
        const fallbackRes = await fetch(endpoint, options);
        if (fallbackRes.ok) {
          return (await fallbackRes.json()) as T;
        }
      } catch {
        // Fall through
      }
    }
    throw err;
  }
}

export const api = {
  getHealth: () => fetchJson<HealthResponse>("/api/health"),
  getSystemStatus: () => fetchJson<SystemStatusResponse>("/api/system/status"),
  getTelemetrySources: () => fetchJson<{ sources: TelemetrySource[]; summary: any }>("/api/telemetry/sources"),
  getTelemetryEvents: (page = 1, limit = 25, source = "all", phase = "all", query = "") =>
    fetchJson<{ page: number; limit: number; total: number; total_pages: number; events: IncidentEvent[] }>(
      `/api/telemetry/events?page=${page}&limit=${limit}&source=${encodeURIComponent(source)}&phase=${encodeURIComponent(phase)}&query=${encodeURIComponent(query)}`
    ),
  getIncidents: (page = 1, limit = 20, split = "test", phase = "all", query = "") =>
    fetchJson<{ page: number; limit: number; split: string; total: number; total_pages: number; incidents: IncidentEvent[] }>(
      `/api/incidents?page=${page}&limit=${limit}&split=${split}&phase=${encodeURIComponent(phase)}&query=${encodeURIComponent(query)}`
    ),
  getIncidentDetail: (id: string) => fetchJson<IncidentEvent>(`/api/incidents/${id}`),
  getDecisions: (page = 1, limit = 20, status = "all") =>
    fetchJson<{ page: number; limit: number; total: number; total_pages: number; decisions: Decision[] }>(
      `/api/decisions?page=${page}&limit=${limit}&status=${status}`
    ),
  getDecisionDetail: (id: string) => fetchJson<Decision>(`/api/decisions/${id}`),
  getSimulations: (params?: { scenario?: string; policy?: string; status?: string; mode?: string }) => {
    const q = new URLSearchParams();
    if (params?.scenario && params.scenario !== "all") q.set("scenario", params.scenario);
    if (params?.policy && params.policy !== "all") q.set("policy", params.policy);
    if (params?.status && params.status !== "all") q.set("status", params.status);
    if (params?.mode && params.mode !== "all") q.set("mode", params.mode);
    const qs = q.toString() ? `?${q.toString()}` : "";
    return fetchJson<{ simulations: SimulationRun[]; total?: number }>(`/api/simulations${qs}`);
  },
  getSimulationDetail: (id: string) => fetchJson<SimulationRun>(`/api/simulations/${id}`),
  compareSimulations: (runA: string, runB: string) =>
    fetchJson<SimulationComparisonResponse>(
      `/api/simulations/compare?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`
    ),
  runSimulation: (scenario: string, steps: number, seed = 42) =>
    fetchJson<SimulationRun>("/api/simulations/run", {
      method: "POST",
      body: JSON.stringify({ scenario, steps, seed }),
    }),
  getEvaluations: () => fetchJson<EvaluationResponse>("/api/evaluations"),
  getResearchExperiments: () => fetchJson<ResearchExperimentsResponse>("/api/research/experiments"),
  getActions: () => fetchJson<{ total: number; actions: (ActionSpaceItem & { tactics?: string[]; class_weight?: number; zero_shot?: boolean })[] }>("/api/actions"),
  evaluateCustomDecision: (text: string, phase?: string, ground_truth_action?: number) =>
    fetchJson<Decision>("/api/decisions/evaluate", {
      method: "POST",
      body: JSON.stringify({ text, phase, ground_truth_action }),
    }),
  getScenarios: () => fetchJson<{ scenarios: ScenarioItem[] }>("/api/scenarios"),
  startSimulation: (payload: StartSimulationPayload) =>
    fetchJson<StartSimulationResponse>("/api/simulations", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  stopSimulation: (runId: string) =>
    fetchJson<{ run_id: string; status: string; message: string }>(`/api/simulations/${runId}/stop`, {
      method: "POST",
    }),
  getSimulationResults: (runId: string) =>
    fetchJson<SimulationResults>(`/api/simulations/${runId}/results`),
  resetSimulations: () =>
    fetchJson<{ status: string; message: string }>("/api/simulations/reset", {
      method: "POST",
    }),
  getSimulationEventStreamUrl: (runId: string) =>
    `${API_BASE}/api/simulations/${runId}/events`,
  getSimulationExportUrl: (runId: string, format: "json" | "csv" = "json") =>
    `${API_BASE}/api/simulations/${runId}/export?format=${format}`,
};
