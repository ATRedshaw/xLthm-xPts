import type {
  ApiDirectory,
  FixtureDetail,
  FixturesResponse,
  Metadata,
  Player,
  PlayersResponse,
} from "../types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    signal,
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { error?: string }
      | null;
    throw new ApiError(payload?.error || `Request failed (${response.status})`, response.status);
  }

  return response.json() as Promise<T>;
}

function queryString(parameters: Record<string, string | number | boolean | undefined>) {
  const query = new URLSearchParams();
  Object.entries(parameters).forEach(([key, value]) => {
    if (value !== undefined && value !== "") query.set(key, String(value));
  });
  const value = query.toString();
  return value ? `?${value}` : "";
}

export const api = {
  health: (signal?: AbortSignal) => request<{ status: string }>("/health", signal),
  metadata: (signal?: AbortSignal) => request<Metadata>("/api/v1/meta", signal),
  directory: (signal?: AbortSignal) => request<ApiDirectory>("/api/v1", signal),
  players: (
    parameters: {
      start_gameweek: number;
      gameweeks: number;
      detail?: "summary" | "standard" | "full";
      limit?: number;
    },
    signal?: AbortSignal,
  ) => request<PlayersResponse>(`/api/v1/players${queryString(parameters)}`, signal),
  player: (
    playerId: number,
    parameters: { start_gameweek: number; gameweeks: number },
    signal?: AbortSignal,
  ) =>
    request<Player>(
      `/api/v1/players/${playerId}${queryString({
        ...parameters,
        detail: "full",
        include_distribution: true,
      })}`,
      signal,
    ),
  fixtures: (
    parameters: { start_gameweek: number; gameweeks: number },
    signal?: AbortSignal,
  ) => request<FixturesResponse>(`/api/v1/fixtures${queryString(parameters)}`, signal),
  fixture: (fixtureId: number, signal?: AbortSignal) =>
    request<FixtureDetail>(
      `/api/v1/fixtures/${fixtureId}${queryString({ detail: "standard" })}`,
      signal,
    ),
};
