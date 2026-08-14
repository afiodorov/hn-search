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
}

pub fn open(path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    // Idempotent: cheap no-op if already present. Lets a direct hn_id lookup
    // (POST /similar) avoid a full table scan without a separate migration step.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hn_id ON doc(hn_id)", [])?;
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

/// Fetch one doc by logical row index (rowid = logical + 1).
pub fn fetch(conn: &Connection, logical: usize) -> Result<Option<Doc>> {
    let mut stmt = conn.prepare_cached(
        "SELECT hn_id, clean_text, author, timestamp, type FROM doc WHERE rowid = ?1",
    )?;
    let row = stmt
        .query_row([(logical + 1) as i64], |r| {
            Ok(Doc {
                hn_id: r.get(0)?,
                clean_text: r.get(1)?,
                author: r.get(2)?,
                timestamp: r.get(3)?,
                doc_type: r.get(4)?,
            })
        })
        .ok();
    Ok(row)
}

/// Fetch one doc (with its logical row index) by hn_id — for direct link/id
/// lookup and "more like this" (POST /similar), neither of which know the rowid.
pub fn fetch_by_hn_id(conn: &Connection, hn_id: &str) -> Result<Option<(usize, Doc)>> {
    let mut stmt = conn.prepare_cached(
        "SELECT rowid, hn_id, clean_text, author, timestamp, type FROM doc WHERE hn_id = ?1",
    )?;
    let row = stmt
        .query_row([hn_id], |r| {
            let rowid: i64 = r.get(0)?;
            Ok((
                (rowid - 1) as usize,
                Doc {
                    hn_id: r.get(1)?,
                    clean_text: r.get(2)?,
                    author: r.get(3)?,
                    timestamp: r.get(4)?,
                    doc_type: r.get(5)?,
                },
            ))
        })
        .ok();
    Ok(row)
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
            "INSERT INTO doc (rowid, hn_id, clean_text, author, timestamp, type) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for (i, d) in docs.iter().enumerate() {
            stmt.execute(rusqlite::params![
                start_rowid + i as i64,
                d.hn_id,
                d.clean_text,
                d.author,
                d.timestamp,
                d.doc_type,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}
