//! SQLite text/metadata store. `doc.rowid` == logical row index + 1, shared by the
//! mmap'd base and the appended tail (rowids continue past the base count).

use anyhow::Result;
use rusqlite::Connection;

pub struct Doc {
    pub hn_id: String,
    pub clean_text: String,
    pub author: String,
    pub timestamp: String,
    pub doc_type: String,
    pub parent_id: Option<String>,
}

/// Idempotent: cheap no-op once the column exists. `ALTER TABLE ADD COLUMN` has
/// no `IF NOT EXISTS` in SQLite, so check `PRAGMA table_info` first — lets this
/// run on every startup instead of needing a separate one-off migration step.
fn ensure_parent_id_column(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(doc)")?;
    let has_column = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .any(|name| name == "parent_id");
    if !has_column {
        conn.execute("ALTER TABLE doc ADD COLUMN parent_id TEXT", [])?;
    }
    Ok(())
}

pub fn open(path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    // Idempotent: cheap no-op if already present. Lets a direct hn_id lookup
    // (POST /similar, POST /docs) avoid a full table scan without a separate
    // migration step.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hn_id ON doc(hn_id)", [])?;
    ensure_parent_id_column(&conn)?;
    Ok(conn)
}

/// Total committed rows (base + tail).
pub fn total_count(conn: &Connection) -> Result<usize> {
    Ok(conn.query_row("SELECT COUNT(*) FROM doc", [], |r| r.get::<_, i64>(0))? as usize)
}

/// Largest hn_id seen (numeric); 0 when empty.
pub fn max_hn_id(conn: &Connection) -> Result<i64> {
    let v: Option<i64> = conn.query_row(
        "SELECT MAX(CAST(hn_id AS INTEGER)) FROM doc",
        [],
        |r| r.get(0),
    )?;
    Ok(v.unwrap_or(0))
}

fn row_to_doc(r: &rusqlite::Row, offset: usize) -> rusqlite::Result<Doc> {
    Ok(Doc {
        hn_id: r.get(offset)?,
        clean_text: r.get(offset + 1)?,
        author: r.get(offset + 2)?,
        timestamp: r.get(offset + 3)?,
        doc_type: r.get(offset + 4)?,
        parent_id: r.get(offset + 5)?,
    })
}

/// Fetch one doc by logical row index (rowid = logical + 1).
pub fn fetch(conn: &Connection, logical: usize) -> Result<Option<Doc>> {
    let mut stmt = conn.prepare_cached(
        "SELECT hn_id, clean_text, author, timestamp, type, parent_id FROM doc WHERE rowid = ?1",
    )?;
    let row = stmt
        .query_row([(logical + 1) as i64], |r| row_to_doc(r, 0))
        .ok();
    Ok(row)
}

/// Fetch one doc (with its logical row index) by hn_id — for direct link/id
/// lookup and "more like this" (POST /similar), neither of which know the rowid.
pub fn fetch_by_hn_id(conn: &Connection, hn_id: &str) -> Result<Option<(usize, Doc)>> {
    let mut stmt = conn.prepare_cached(
        "SELECT rowid, hn_id, clean_text, author, timestamp, type, parent_id FROM doc WHERE hn_id = ?1",
    )?;
    let row = stmt
        .query_row([hn_id], |r| {
            let rowid: i64 = r.get(0)?;
            Ok(((rowid - 1) as usize, row_to_doc(r, 1)?))
        })
        .ok();
    Ok(row)
}

/// Batch fetch docs by hn_id (POST /docs) — e.g. resolving a set of comments'
/// parent_ids to their own text/author/timestamp in one round trip. Missing
/// ids are silently skipped, not an error.
pub fn fetch_by_hn_ids(conn: &Connection, hn_ids: &[String]) -> Result<Vec<Doc>> {
    let mut stmt = conn.prepare_cached(
        "SELECT hn_id, clean_text, author, timestamp, type, parent_id FROM doc WHERE hn_id = ?1",
    )?;
    let mut out = Vec::with_capacity(hn_ids.len());
    for id in hn_ids {
        if let Some(doc) = stmt.query_row([id], |r| row_to_doc(r, 0)).ok() {
            out.push(doc);
        }
    }
    Ok(out)
}

/// Which of the given hn_ids already exist (for append dedup).
pub fn existing_hn_ids(conn: &Connection, ids: &[String]) -> Result<std::collections::HashSet<String>> {
    let mut set = std::collections::HashSet::new();
    let mut stmt = conn.prepare_cached("SELECT 1 FROM doc WHERE hn_id = ?1 LIMIT 1")?;
    for id in ids {
        if stmt.exists([id])? {
            set.insert(id.clone());
        }
    }
    Ok(set)
}

/// Insert tail docs starting at the given rowid (== first logical index + 1).
/// Caller guarantees rows are new (deduped) and rowids are contiguous.
pub fn insert_tail(conn: &mut Connection, start_rowid: i64, docs: &[Doc]) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO doc (rowid, hn_id, clean_text, author, timestamp, type, parent_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for (i, d) in docs.iter().enumerate() {
            stmt.execute(rusqlite::params![
                start_rowid + i as i64,
                d.hn_id,
                d.clean_text,
                d.author,
                d.timestamp,
                d.doc_type,
                d.parent_id,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}
