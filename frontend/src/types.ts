export type Position = "GK" | "DEF" | "MID" | "FWD";

export interface ResponseMeta {
  season: string;
  generated_at: string;
  start_gameweek: number;
  end_gameweek: number;
  detail: string;
  count: number;
  limit?: number;
  offset?: number;
}

export interface OutcomeProbabilities {
  negative_points: number;
  zero_points: number;
  two_plus_points: number;
  five_plus_points: number;
  ten_plus_points: number;
  percentiles: Record<string, number>;
  points_distribution?: Record<string, number>;
}

export interface FixtureProjection {
  fixture: number;
  opponent: string;
  is_home: boolean;
  xpts: number;
  xmins: number;
  action_probabilities?: Record<string, number>;
  expected_actions?: Record<string, number>;
  xpts_breakdown?: Record<string, number>;
  outcome_probabilities?: OutcomeProbabilities;
}

export interface GameweekProjection {
  gameweek: number;
  xpts: number;
  xmins: number;
  fixture_projections?: FixtureProjection[];
  outcome_probabilities?: OutcomeProbabilities;
}

export interface Player {
  id: number;
  code?: number;
  name: string;
  position: Position;
  team: string;
  team_id?: number;
  price: number;
  selected_by: number;
  status?: string;
  availability_probability?: number;
  news?: string;
  future_points: GameweekProjection[];
  total_xpts: number;
}

export interface PlayersResponse {
  meta: ResponseMeta;
  players: Player[];
}

export interface FixtureForecast {
  expected_goals: { home: number; away: number };
  result_probabilities: {
    home_win: number;
    draw: number;
    away_win: number;
  };
  clean_sheet_probabilities: { home: number; away: number };
}

export interface Fixture {
  fixture: number;
  gameweek: number;
  kickoff_time: string;
  home_team: string;
  away_team: string;
  home_team_name: string;
  away_team_name: string;
  forecast: FixtureForecast;
}

export interface FixturePlayer extends Omit<Player, "future_points" | "total_xpts"> {
  projection: FixtureProjection;
}

export interface FixtureDetail extends Fixture {
  players: FixturePlayer[];
}

export interface FixturesResponse {
  meta: ResponseMeta;
  fixtures: Fixture[];
}

export interface ModelMetadata {
  model_type: string;
  artifact_version: number;
  trained_at: string;
  feature_profile: string | null;
}

export interface Metadata {
  generated_at: string;
  data_retrieved_at: string;
  season: string;
  ruleset: string;
  simulation_count: number;
  random_state: number;
  models: Record<string, ModelMetadata>;
  coverage: {
    players: number;
    fixtures: number;
    gameweeks: number[];
    skipped_fixtures: number[];
  };
  quality_checks: Record<string, number>;
  methodology: Record<string, string>;
  limitations: string[];
}

export interface ApiDirectory {
  name: string;
  version: string;
  defaults: Record<string, unknown>;
  endpoints: Record<
    string,
    { method: string; description: string; parameters: string[] }
  >;
  parameters: Record<string, Record<string, unknown>>;
  outcome_probabilities: Record<string, unknown>;
  examples: string[];
}

export type ViewName = "players" | "fixtures" | "model";
