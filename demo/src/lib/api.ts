export interface DemoItem {
  item_id: number;
  asin: string;
  title: string;
  category: string;
  tag: string;
}

export interface RecommendationItem extends DemoItem {
  rank: number;
  score: number;
  score_percent: number;
}

export interface RecommendationResponse {
  user_id: string;
  history: DemoItem[];
  cloud: {
    candidate_ids: number[];
    candidate_count: number;
    embedding_dim: number;
    embedding_source: string;
    catalog_items: number;
    source_summary: {
      graph_neighbors: number;
      vector_neighbors: number;
      fallback_items: number;
    };
  };
  inference: {
    model: string;
    runtime: string;
    runtime_detail: string;
    neighbor_count: number;
    candidate_count: number;
    user_vector_preview: number[];
    candidate_vector_preview: number[];
    dot_score_preview: number;
  };
  recommendations: RecommendationItem[];
}

const API_BASE_URL = import.meta.env.VITE_DCCL_API_URL ?? "http://localhost:8000";

export async function fetchRecommendations(
  userId?: string,
  topK = 5,
  targetK = 50,
): Promise<RecommendationResponse> {
  const url = new URL("/api/v1/recommendations/", API_BASE_URL);
  if (userId) {
    url.searchParams.set("user_id", userId);
  }
  url.searchParams.set("top_k", String(topK));
  url.searchParams.set("target_k", String(targetK));

  const response = await fetch(url);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}
