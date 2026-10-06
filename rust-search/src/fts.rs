//! Full-text index over `doc.clean_text`, for exact keyword questions: how many
//! comments mention a term, when it first appeared, how that changed by month.
//!
//! A contentless FTS5 table (`content=''`) in its own file, `fts.sqlite`: it
//! stores only the term → rowid postings, keyed by the same rowids as `doc`, so
//! text and timestamps still come from `docs.sqlite`. Being a separate file is
//! the point: it is built off-box by `misc/build_fts.py` (the build needs RAM
//! and CPU the search box doesn't have), shipped next to the other artifacts,
//! and if it is missing the service runs as before with `/keyword` returning 503.
//!
//! The file stops at whatever rowid it was built up to. `sync` indexes every
//! row after that, from `docs.sqlite`; it runs at startup and after each
//! `/append`, so the index catches up on its own after a ship and after a crash
//! between the doc commit and the index insert. The tokenizer is part of the
//! table definition stored in the file, so rows indexed here are split exactly
//! like the ones indexed by the build.

use std::ops::Range;

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;

use crate::db;

/// Must match `misc/build_fts.py`. `detail=full` keeps positions, for phrases
/// ("borrow checker"); `columnsize=0` drops per-row sizes, which only bm25
/// ranking needs, and we never rank.
pub const CREATE: &str = "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(\
    clean_text, content='', columnsize=0, detail=full, tokenize='unicode61')";

/// The highest rowid indexed. A contentless table can't be scanned without a
/// MATCH ("table does not support scanning"), so MAX(rowid) on it is an error;
/// this one-row table is written in the same transaction as the postings.
pub const CREATE_META: &str =
    "CREATE TABLE IF NOT EXISTS fts_meta (id INTEGER PRIMARY KEY CHECK (id = 1), max_rowid INTEGER NOT NULL)";

pub fn open(path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    init(&conn)?;
    Ok(conn)
}

fn init(conn: &Connection) -> Result<()> {
    conn.execute(CREATE, [])?;
    conn.execute(CREATE_META, [])?;
    conn.execute("INSERT OR IGNORE INTO fts_meta (id, max_rowid) VALUES (1, 0)", [])?;
    Ok(())
}

/// Highest rowid indexed; 0 when empty.
pub fn max_rowid(fts: &Connection) -> Result<i64> {
    Ok(fts.query_row("SELECT max_rowid FROM fts_meta WHERE id = 1", [], |r| r.get(0))?)
}

/// Index every committed doc row after the highest one already indexed, in
/// batches. Returns how many rows were added.
pub fn sync(fts: &mut Connection, docs: &Connection) -> Result<usize> {
    const BATCH: i64 = 10_000;
    let mut next = max_rowid(fts)? + 1;
    let mut added = 0;
    let mut read = docs.prepare_cached(
        "SELECT rowid, clean_text FROM doc WHERE rowid >= ?1 ORDER BY rowid LIMIT ?2",
    )?;
    loop {
        let rows: Vec<(i64, String)> = read
            .query_map([next, BATCH], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        let Some(&(last, _)) = rows.last() else {
            return Ok(added);
        };
        let tx = fts.transaction()?;
        {
            let mut ins = tx.prepare_cached("INSERT INTO fts (rowid, clean_text) VALUES (?1, ?2)")?;
            for (rowid, text) in &rows {
                ins.execute(rusqlite::params![rowid, text])?;
            }
            tx.execute("UPDATE fts_meta SET max_rowid = ?1 WHERE id = 1", [last])?;
        }
        tx.commit()?;
        added += rows.len();
        next = last + 1;
    }
}

/// The term as one FTS5 phrase: every token must appear, adjacent and in
/// order. Quoting makes the whole input literal, so FTS5 operators in it
/// (AND, NEAR, *, column filters) are searched as words, never interpreted.
pub fn phrase(term: &str) -> String {
    format!("\"{}\"", term.replace('"', "\"\""))
}

/// Call `f` with each matching rowid (1-based, as stored) inside `lo..=hi`,
/// ascending, without collecting them: a common word matches millions.
pub fn for_each_match(
    fts: &Connection,
    term: &str,
    lo: i64,
    hi: i64,
    mut f: impl FnMut(i64),
) -> Result<()> {
    let mut stmt = fts.prepare_cached(
        "SELECT rowid FROM fts WHERE fts MATCH ?1 AND rowid BETWEEN ?2 AND ?3 ORDER BY rowid",
    )?;
    let mut rows = stmt.query(rusqlite::params![phrase(term), lo, hi])?;
    while let Some(r) = rows.next()? {
        f(r.get(0)?);
    }
    Ok(())
}

#[derive(Serialize, Debug, PartialEq)]
pub struct Month {
    /// "YYYY-MM"
    pub month: String,
    /// Comments in the month that contain the term.
    pub count: usize,
    /// All comments in the month (inside the requested window), so a count
    /// can be read as a share: the archive's volume varies month to month.
    pub rows: usize,
}

pub struct Stats {
    pub count: usize,
    pub rows: usize,
    /// Logical row indexes of the earliest matches, oldest first.
    pub first: Vec<usize>,
    pub months: Vec<Month>,
}

fn next_month(ym: &str) -> String {
    let (y, m): (i32, u32) = (ym[..4].parse().unwrap_or(0), ym[5..7].parse().unwrap_or(1));
    if m == 12 {
        format!("{:04}-01", y + 1)
    } else {
        format!("{y:04}-{:02}", m + 1)
    }
}

/// Count matches of `term` among logical rows `rows`, per calendar month, and
/// collect the first `k`. Months are contiguous row ranges (rows are in time
/// order), so their edges come from the same binary search as `/search`'s
/// windows and every match is bucketed by rowid alone, with no per-match
/// timestamp lookup.
pub fn stats(
    fts: &Connection,
    docs: &Connection,
    total: usize,
    term: &str,
    rows: Range<usize>,
    k: usize,
) -> Result<Stats> {
    let mut out = Stats { count: 0, rows: rows.len(), first: Vec::new(), months: Vec::new() };
    if rows.is_empty() {
        return Ok(out);
    }
    let ts = |i: usize| -> Result<String> {
        Ok(db::fetch(docs, i)?.map(|d| d.timestamp).unwrap_or_default())
    };
    let (first_ts, last_ts) = (ts(rows.start)?, ts(rows.end - 1)?);
    if first_ts.len() < 7 || last_ts.len() < 7 {
        anyhow::bail!("unexpected timestamp format: {first_ts:?} / {last_ts:?}");
    }
    // (month, first logical row of the month), clipped to `rows`.
    let mut edges: Vec<(String, usize)> = Vec::new();
    let mut month = first_ts[..7].to_string();
    while month.as_str() <= &last_ts[..7] {
        let start = db::time_range(docs, total, Some(&month), None)?.start;
        edges.push((month.clone(), start.clamp(rows.start, rows.end)));
        month = next_month(&month);
    }
    edges[0].1 = rows.start;
    let ends: Vec<usize> = edges.iter().skip(1).map(|e| e.1).chain([rows.end]).collect();
    let mut counts = vec![0usize; edges.len()];

    let mut bucket = 0;
    for_each_match(fts, term, rows.start as i64 + 1, rows.end as i64, |rowid| {
        let logical = (rowid - 1) as usize;
        while bucket + 1 < edges.len() && logical >= edges[bucket + 1].1 {
            bucket += 1;
        }
        counts[bucket] += 1;
        out.count += 1;
        if out.first.len() < k {
            out.first.push(logical);
        }
    })?;

    out.months = edges
        .into_iter()
        .zip(ends)
        .zip(counts)
        .map(|(((month, start), end), count)| Month { month, count, rows: end - start })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matching_rowids(fts: &Connection, term: &str, lo: i64, hi: i64) -> Result<Vec<i64>> {
        let mut ids = Vec::new();
        for_each_match(fts, term, lo, hi, |id| ids.push(id))?;
        Ok(ids)
    }

    fn docs(texts: &[&str]) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE doc (rowid INTEGER PRIMARY KEY, hn_id TEXT, clean_text TEXT, \
             author TEXT, timestamp TEXT, type TEXT, parent_id TEXT)",
            [],
        )
        .unwrap();
        for (i, t) in texts.iter().enumerate() {
            conn.execute(
                "INSERT INTO doc (rowid, hn_id, clean_text, author, timestamp, type) \
                 VALUES (?1, ?1, ?2, 'a', '2023-01-01', 'comment')",
                rusqlite::params![i as i64 + 1, t],
            )
            .unwrap();
        }
        conn
    }

    fn fts() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init(&conn).unwrap();
        conn
    }

    #[test]
    fn the_bundled_sqlite_has_fts5_and_sync_catches_up() {
        let d = docs(&["I asked ChatGPT", "no mention", "chatgpt's answer was wrong"]);
        let mut f = fts();
        assert_eq!(sync(&mut f, &d).unwrap(), 3);
        assert_eq!(sync(&mut f, &d).unwrap(), 0, "a second sync has nothing to add");
        d.execute(
            "INSERT INTO doc (rowid, hn_id, clean_text) VALUES (4, '4', 'ChatGPT again')",
            [],
        )
        .unwrap();
        assert_eq!(sync(&mut f, &d).unwrap(), 1);
        assert_eq!(matching_rowids(&f, "chatgpt", 1, 4).unwrap(), vec![1, 3, 4]);
        assert_eq!(matching_rowids(&f, "chatgpt", 2, 3).unwrap(), vec![3]);
    }

    #[test]
    fn terms_are_words_and_phrases_not_substrings_or_syntax() {
        let d = docs(&[
            "the borrow checker hates me",
            "checker borrow",
            "I trust it",
            "NEAR OR AND *",
            "say \"hi\" there",
        ]);
        let mut f = fts();
        sync(&mut f, &d).unwrap();
        assert_eq!(matching_rowids(&f, "Borrow Checker", 1, 9).unwrap(), vec![1]);
        assert!(matching_rowids(&f, "rust", 1, 9).unwrap().is_empty(), "trust is not rust");
        // Operators in the input are plain words, never query syntax.
        assert_eq!(matching_rowids(&f, "NEAR OR", 1, 9).unwrap(), vec![4]);
        assert_eq!(matching_rowids(&f, "say \"hi", 1, 9).unwrap(), vec![5]);
    }

    #[test]
    fn stats_bucket_by_month_and_respect_the_window() {
        let d = docs(&[]);
        let rows = [
            ("2023-01-05", "chatgpt is new"),
            ("2023-01-20", "nothing"),
            ("2023-02-01", "chatgpt again"),
            ("2023-02-11", "and chatgpt"),
            ("2023-04-02", "skipped a month; chatgpt"),
            ("2023-04-03", "nope"),
        ];
        for (i, (ts, text)) in rows.iter().enumerate() {
            d.execute(
                "INSERT INTO doc (rowid, hn_id, clean_text, author, timestamp, type) \
                 VALUES (?1, ?1, ?2, 'a', ?3, 'comment')",
                rusqlite::params![i as i64 + 1, text, format!("{ts} 10:00:00+00:00")],
            )
            .unwrap();
        }
        let mut f = fts();
        sync(&mut f, &d).unwrap();

        let all = stats(&f, &d, 6, "ChatGPT", 0..6, 2).unwrap();
        assert_eq!((all.count, all.rows, all.first.clone()), (4, 6, vec![0, 2]));
        let m = |month: &str, count, rows| Month { month: month.into(), count, rows };
        assert_eq!(
            all.months,
            vec![m("2023-01", 1, 2), m("2023-02", 2, 2), m("2023-03", 0, 0), m("2023-04", 1, 2)]
        );

        // A window that starts mid-January and ends mid-February.
        let w = db::time_range(&d, 6, Some("2023-01-10"), Some("2023-02-05")).unwrap();
        let part = stats(&f, &d, 6, "chatgpt", w, 5).unwrap();
        assert_eq!((part.count, part.rows, part.first), (1, 2, vec![2]));
        assert_eq!(part.months, vec![m("2023-01", 0, 1), m("2023-02", 1, 1)]);

        assert_eq!(stats(&f, &d, 6, "chatgpt", 3..3, 5).unwrap().count, 0);
        assert_eq!(next_month("2023-12"), "2024-01");
    }

    /// A file built by misc/build_fts.py (Python's SQLite) must open and answer
    /// here (rusqlite's bundled SQLite). Run with
    /// `FTS_FILE=path cargo test -- --ignored built_file`.
    #[test]
    #[ignore]
    fn a_built_file_reads_back() {
        let path = std::env::var("FTS_FILE").expect("set FTS_FILE");
        let f = open(std::path::Path::new(&path)).unwrap();
        let max = max_rowid(&f).unwrap();
        let hits = matching_rowids(&f, "chatgpt", 1, max).unwrap();
        eprintln!("sqlite {} max_rowid {max} chatgpt {}", rusqlite::version(), hits.len());
        assert!(max > 0 && !hits.is_empty());
    }
}
