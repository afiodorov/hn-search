//! HTTP vector-search service: binary Hamming shortlist + f16 cosine rerank over an
//! mmap'd base, with an appendable tail for daily online updates.
//!
//! Endpoints (all but /health require `Authorization: Bearer $HN_SEARCH_TOKEN` when
//! the token env is set):
//!   GET  /health  -> "ok"
//!   POST /search  {embedding:[f32;768], k?, shortlist?, time_after?, time_before?} -> [SearchHit]
//!   POST /similar {hn_id, k?} -> [SearchHit]  (reuses the doc's own stored vector)
//!   POST /docs    {hn_ids:[...]} -> [DocHit]  (batch text/metadata lookup, e.g. resolving parent_ids)
//!   POST /append  {rows:[{hn_id, clean_text, author, timestamp, type, embedding, parent_id?}]}
//!   GET  /max_id  -> {max_id}

mod db;
mod index;
mod quantize;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use axum::{
    extract::{DefaultBodyLimit, Json, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use index::{Base, Tail};
use quantize::DIM;

struct AppState {
    base: Base,
    tail: RwLock<Tail>,
    db: Mutex<Connection>,
    read_token: Option<String>,
    admin_token: Option<String>,
    shortlist: usize,
}

#[derive(Deserialize)]
struct SearchReq {
    embedding: Vec<f32>,
    k: Option<usize>,
    shortlist: Option<usize>,
    /// Inclusive ISO8601 bounds on `timestamp` (string-compared — valid since the
    /// stored format is zero-padded ISO8601 with the date first). Filtering
    /// happens after the usual Hamming shortlist + cosine rerank, so when either
    /// bound is set we ask index::search for more candidates than `k` up front
    /// to leave enough survivors after the filter.
    time_after: Option<String>,
    time_before: Option<String>,
}

#[derive(Deserialize)]
struct SimilarReq {
    hn_id: String,
    k: Option<usize>,
}

#[derive(Deserialize)]
struct DocsReq {
    hn_ids: Vec<String>,
}

#[derive(Serialize)]
struct SearchHit {
    id: String,
    clean_text: String,
    author: String,
    timestamp: String,
    #[serde(rename = "type")]
    doc_type: String,
    distance: f32,
}

#[derive(Serialize)]
struct DocHit {
    id: String,
    clean_text: String,
    author: String,
    timestamp: String,
    #[serde(rename = "type")]
    doc_type: String,
    parent_id: Option<String>,
}

#[derive(Deserialize)]
struct AppendRow {
    hn_id: String,
    clean_text: String,
    author: String,
    timestamp: String,
    #[serde(rename = "type")]
    doc_type: String,
    embedding: Vec<f32>,
    #[serde(default)]
    parent_id: Option<String>,
}

#[derive(Deserialize)]
struct AppendReq {
    rows: Vec<AppendRow>,
}

#[derive(Serialize)]
struct AppendResp {
    appended: usize,
    skipped: usize,
    max_id: i64,
}

type ApiError = (StatusCode, String);

fn err(code: StatusCode, msg: impl Into<String>) -> ApiError {
    (code, msg.into())
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

fn auth_enabled(state: &AppState) -> bool {
    state.read_token.is_some() || state.admin_token.is_some()
}

/// `/search`: accepts the read token OR the admin token (admin is a superset).
fn require_read(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if !auth_enabled(state) {
        return Ok(()); // no tokens configured → open (local dev)
    }
    let got = bearer(headers);
    if got.is_some() && (got == state.read_token.as_deref() || got == state.admin_token.as_deref()) {
        Ok(())
    } else {
        Err(err(StatusCode::UNAUTHORIZED, "bad or missing bearer token"))
    }
}

/// `/append`, `/max_id`: admin token only. If no admin token is configured (but auth
/// is otherwise on), writes are impossible by design — a safe default for a public box.
fn require_admin(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if !auth_enabled(state) {
        return Ok(());
    }
    let got = bearer(headers);
    if got.is_some() && got == state.admin_token.as_deref() {
        Ok(())
    } else {
        Err(err(StatusCode::UNAUTHORIZED, "admin token required"))
    }
}

async fn health() -> &'static str {
    "ok"
}

async fn max_id(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    require_admin(&state, &headers)?;
    let id = state.tail.read().unwrap().max_id;
    Ok(Json(serde_json::json!({ "max_id": id })))
}

async fn search(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<SearchReq>,
) -> Result<impl IntoResponse, ApiError> {
    require_read(&state, &headers)?;
    if req.embedding.len() != DIM {
        return Err(err(
            StatusCode::BAD_REQUEST,
            format!("embedding must have {DIM} dims, got {}", req.embedding.len()),
        ));
    }
    let k = req.k.unwrap_or(10);
    let has_time_filter = req.time_after.is_some() || req.time_before.is_some();
    // Filtering happens after rerank truncates to fetch_k, so over-fetch when a
    // time bound is set to leave enough survivors; plain searches are unaffected.
    let fetch_k = if has_time_filter { k * 5 } else { k };
    let shortlist = req.shortlist.unwrap_or(state.shortlist).max(fetch_k);
    let time_after = req.time_after.clone();
    let time_before = req.time_before.clone();

    let hits = tokio::task::spawn_blocking(move || -> Result<Vec<SearchHit>, String> {
        let tail = state.tail.read().unwrap();
        let scored = index::search(&state.base, &tail, &req.embedding, shortlist, fetch_k);
        drop(tail);
        let conn = state.db.lock().unwrap();
        let mut out = Vec::with_capacity(k);
        for (idx, dist) in scored {
            if out.len() == k {
                break;
            }
            if let Some(d) = db::fetch(&conn, idx).map_err(|e| e.to_string())? {
                if let Some(after) = &time_after {
                    if &d.timestamp < after {
                        continue;
                    }
                }
                if let Some(before) = &time_before {
                    if &d.timestamp > before {
                        continue;
                    }
                }
                out.push(SearchHit {
                    id: d.hn_id,
                    clean_text: d.clean_text,
                    author: d.author,
                    timestamp: d.timestamp,
                    doc_type: d.doc_type,
                    distance: dist,
                });
            }
        }
        Ok(out)
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(hits))
}

async fn similar(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<SimilarReq>,
) -> Result<impl IntoResponse, ApiError> {
    require_read(&state, &headers)?;
    let k = req.k.unwrap_or(10);

    let hits = tokio::task::spawn_blocking(move || -> Result<Option<Vec<SearchHit>>, String> {
        let conn = state.db.lock().unwrap();
        let found = db::fetch_by_hn_id(&conn, &req.hn_id).map_err(|e| e.to_string())?;
        let Some((source_idx, _)) = found else {
            return Ok(None);
        };
        drop(conn);

        let tail = state.tail.read().unwrap();
        let query = index::get_vector(&state.base, &tail, source_idx);
        // +1 candidate to absorb the source document itself, which will always
        // be its own nearest neighbor at distance ~0.
        let shortlist = state.shortlist.max(k + 1);
        let scored = index::search(&state.base, &tail, &query, shortlist, k + 1);
        drop(tail);

        let conn = state.db.lock().unwrap();
        let mut out = Vec::with_capacity(k);
        for (idx, dist) in scored {
            if out.len() == k {
                break;
            }
            if idx == source_idx {
                continue;
            }
            if let Some(d) = db::fetch(&conn, idx).map_err(|e| e.to_string())? {
                out.push(SearchHit {
                    id: d.hn_id,
                    clean_text: d.clean_text,
                    author: d.author,
                    timestamp: d.timestamp,
                    doc_type: d.doc_type,
                    distance: dist,
                });
            }
        }
        Ok(Some(out))
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    match hits {
        Some(hits) => Ok(Json(hits)),
        None => Err(err(StatusCode::NOT_FOUND, "hn_id not found")),
    }
}

async fn docs(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<DocsReq>,
) -> Result<impl IntoResponse, ApiError> {
    require_read(&state, &headers)?;

    let hits = tokio::task::spawn_blocking(move || -> Result<Vec<DocHit>, String> {
        let conn = state.db.lock().unwrap();
        let found = db::fetch_by_hn_ids(&conn, &req.hn_ids).map_err(|e| e.to_string())?;
        Ok(found
            .into_iter()
            .map(|d| DocHit {
                id: d.hn_id,
                clean_text: d.clean_text,
                author: d.author,
                timestamp: d.timestamp,
                doc_type: d.doc_type,
                parent_id: d.parent_id,
            })
            .collect())
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(hits))
}

async fn append(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<AppendReq>,
) -> Result<impl IntoResponse, ApiError> {
    require_admin(&state, &headers)?;
    for r in &req.rows {
        if r.embedding.len() != DIM {
            return Err(err(
                StatusCode::BAD_REQUEST,
                format!("embedding for {} must have {DIM} dims", r.hn_id),
            ));
        }
    }

    let resp = tokio::task::spawn_blocking(move || -> Result<AppendResp, String> {
        // Hold the tail write lock across the whole op so searches never see tail
        // vectors whose SQLite docs aren't committed yet.
        let mut tail = state.tail.write().unwrap();
        let mut conn = state.db.lock().unwrap();

        let incoming_ids: Vec<String> = req.rows.iter().map(|r| r.hn_id.clone()).collect();
        let mut seen = db::existing_hn_ids(&conn, &incoming_ids).map_err(|e| e.to_string())?;

        let mut docs = Vec::new();
        let mut vecs = Vec::new();
        let mut new_max = tail.max_id;
        for r in req.rows {
            if !seen.insert(r.hn_id.clone()) {
                continue; // already present (DB) or duplicate within batch
            }
            if let Ok(n) = r.hn_id.parse::<i64>() {
                new_max = new_max.max(n);
            }
            docs.push(db::Doc {
                hn_id: r.hn_id,
                clean_text: r.clean_text,
                author: r.author,
                timestamp: r.timestamp,
                doc_type: r.doc_type,
                parent_id: r.parent_id,
            });
            vecs.push(r.embedding);
        }

        let appended = docs.len();
        let skipped = incoming_ids.len() - appended;
        if appended > 0 {
            let start_rowid = (state.base.count + tail.count) as i64 + 1;
            tail.append(&vecs, new_max).map_err(|e| e.to_string())?;
            db::insert_tail(&mut conn, start_rowid, &docs).map_err(|e| e.to_string())?;
        }
        Ok(AppendResp {
            appended,
            skipped,
            max_id: tail.max_id,
        })
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(resp))
}

#[derive(Deserialize)]
struct Meta {
    count: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::var("ARTIFACT_DIR").unwrap_or_else(|_| "artifacts".into()));
    let read_token = std::env::var("HN_SEARCH_TOKEN").ok().filter(|s| !s.is_empty());
    let admin_token = std::env::var("HN_SEARCH_ADMIN_TOKEN").ok().filter(|s| !s.is_empty());
    let shortlist: usize = std::env::var("HN_SHORTLIST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8001);

    let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let conn = db::open(&dir.join("docs.sqlite"))?;
    let sqlite_total = db::total_count(&conn)?;
    let start_max_id = db::max_hn_id(&conn)?;

    let base = Base::open(&dir, meta.count)?;
    let tail = Tail::load(&dir, meta.count, sqlite_total, start_max_id)?;
    eprintln!(
        "loaded base={} tail={} max_id={} shortlist={} read_auth={} admin_auth={}",
        base.count,
        tail.count,
        start_max_id,
        shortlist,
        read_token.is_some(),
        admin_token.is_some()
    );

    let state = Arc::new(AppState {
        base,
        tail: RwLock::new(tail),
        db: Mutex::new(conn),
        read_token,
        admin_token,
        shortlist,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/search", post(search))
        .route("/similar", post(similar))
        .route("/docs", post(docs))
        // axum defaults request bodies to 2MB; /append batches are several MB of
        // JSON embeddings, so lift the cap to match Caddy's 64MB.
        .route(
            "/append",
            post(append).layer(DefaultBodyLimit::max(64 * 1024 * 1024)),
        )
        .route("/max_id", get(max_id))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
