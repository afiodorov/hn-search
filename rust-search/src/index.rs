//! Base (mmap'd, immutable) + tail (appendable) vector index.
//!
//! Logical row index addresses both: `0..base.count` are mmap base rows,
//! `base.count..base.count+tail.count` are tail rows. `db::Doc` rowids == logical+1.
//! Stage 1 is a parallel Hamming scan of base + a serial scan of the (small) tail;
//! stage 2 reranks the shortlist by exact cosine over the f16 vectors.

use crate::quantize::{hamming, quantize, CODE_BYTES, DIM};
use anyhow::{Context, Result};
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::fs::OpenOptions;
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};

const F16_ROW: usize = DIM * 2; // 1536 bytes

fn map(path: &Path) -> Result<Mmap> {
    let f = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
    Ok(unsafe { Mmap::map(&f)? })
}

pub struct Base {
    codes: Mmap,
    f16: Mmap,
    pub count: usize,
}

impl Base {
    pub fn open(dir: &Path, count: usize) -> Result<Base> {
        let codes = map(&dir.join("codes.bin"))?;
        let f16 = map(&dir.join("rerank_f16.bin"))?;
        anyhow::ensure!(
            codes.len() == count * CODE_BYTES,
            "codes.bin is {} bytes, expected {}",
            codes.len(),
            count * CODE_BYTES
        );
        anyhow::ensure!(
            f16.len() == count * F16_ROW,
            "rerank_f16.bin is {} bytes, expected {}",
            f16.len(),
            count * F16_ROW
        );
        // Warm the hot codes into RAM so the first query isn't paging from disk.
        let mut sink = 0u64;
        for b in codes.iter().step_by(4096) {
            sink = sink.wrapping_add(*b as u64);
        }
        std::hint::black_box(sink);
        Ok(Base { codes, f16, count })
    }
    #[inline]
    fn code(&self, i: usize) -> &[u8] {
        &self.codes[i * CODE_BYTES..(i + 1) * CODE_BYTES]
    }
    #[inline]
    fn f16(&self, i: usize) -> &[u8] {
        &self.f16[i * F16_ROW..(i + 1) * F16_ROW]
    }
}

pub struct Tail {
    codes: Vec<u8>,
    f16: Vec<u8>,
    pub count: usize,
    pub max_id: i64,
    /// ISO8601 timestamp of the newest doc (base + tail); "" if the store is empty.
    pub latest_timestamp: String,
    codes_path: PathBuf,
    f16_path: PathBuf,
}

impl Tail {
    /// Load tail segment, reconciling vector files against the SQLite row count.
    /// Append order is files-first then SQLite commit, so files may hold orphan
    /// rows after a crash — those are trimmed to match committed `sqlite_total`.
    pub fn load(
        dir: &Path,
        base_count: usize,
        sqlite_total: usize,
        max_id: i64,
        latest_timestamp: String,
    ) -> Result<Tail> {
        let codes_path = dir.join("tail_codes.bin");
        let f16_path = dir.join("tail_f16.bin");
        let mut codes = std::fs::read(&codes_path).unwrap_or_default();
        let mut f16 = std::fs::read(&f16_path).unwrap_or_default();

        let want = sqlite_total.saturating_sub(base_count);
        let file_rows = codes.len() / CODE_BYTES;
        anyhow::ensure!(
            file_rows >= want,
            "tail vector files hold {file_rows} rows but SQLite expects {want}"
        );
        codes.truncate(want * CODE_BYTES);
        f16.truncate(want * F16_ROW);
        if file_rows != want {
            // Drop orphan vectors so disk matches committed rows.
            std::fs::write(&codes_path, &codes)?;
            std::fs::write(&f16_path, &f16)?;
        }
        Ok(Tail {
            codes,
            f16,
            count: want,
            max_id,
            latest_timestamp,
            codes_path,
            f16_path,
        })
    }

    #[inline]
    fn code(&self, t: usize) -> &[u8] {
        &self.codes[t * CODE_BYTES..(t + 1) * CODE_BYTES]
    }
    #[inline]
    fn f16(&self, t: usize) -> &[u8] {
        &self.f16[t * F16_ROW..(t + 1) * F16_ROW]
    }

    /// Append vectors (files-first + fsync, then in-memory). Returns nothing;
    /// the caller commits the matching SQLite rows afterwards.
    pub fn append(
        &mut self,
        vecs: &[Vec<f32>],
        new_max_id: i64,
        new_latest_timestamp: &str,
    ) -> Result<()> {
        let mut code_buf = Vec::with_capacity(vecs.len() * CODE_BYTES);
        let mut f16_buf = Vec::with_capacity(vecs.len() * F16_ROW);
        for v in vecs {
            code_buf.extend_from_slice(&quantize(v));
            for &x in v {
                f16_buf.extend_from_slice(&f16::from_f32(x).to_le_bytes());
            }
        }
        append_sync(&self.codes_path, &code_buf)?;
        append_sync(&self.f16_path, &f16_buf)?;
        self.codes.extend_from_slice(&code_buf);
        self.f16.extend_from_slice(&f16_buf);
        self.count += vecs.len();
        self.max_id = self.max_id.max(new_max_id);
        if new_latest_timestamp > self.latest_timestamp.as_str() {
            self.latest_timestamp = new_latest_timestamp.to_string();
        }
        Ok(())
    }
}

fn append_sync(path: &Path, buf: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(buf)?;
    f.sync_all()?;
    Ok(())
}

#[inline]
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    (0..DIM)
        .map(|i| f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]).to_f32())
        .collect()
}

#[inline]
fn push_bounded(heap: &mut BinaryHeap<(u32, usize)>, cap: usize, d: u32, idx: usize) {
    if heap.len() < cap {
        heap.push((d, idx));
    } else if let Some(&(worst, _)) = heap.peek() {
        if d < worst {
            heap.pop();
            heap.push((d, idx));
        }
    }
}

fn scan_base(base: &Base, qcode: &[u8], shortlist: usize, rows: Range<usize>) -> BinaryHeap<(u32, usize)> {
    (rows.start.min(base.count)..rows.end.min(base.count))
        .into_par_iter()
        .fold(BinaryHeap::new, |mut h, i| {
            push_bounded(&mut h, shortlist, hamming(qcode, base.code(i)), i);
            h
        })
        .reduce(BinaryHeap::new, |mut a, b| {
            for (d, i) in b {
                push_bounded(&mut a, shortlist, d, i);
            }
            a
        })
}

/// Read and decode the stored vector for a logical row index (base or tail).
/// Lets POST /similar reuse a document's own already-computed embedding as a
/// query vector — no reembedding, just the same bytes rerank already reads.
pub fn get_vector(base: &Base, tail: &Tail, idx: usize) -> Vec<f32> {
    let bytes = if idx < base.count {
        base.f16(idx)
    } else {
        tail.f16(idx - base.count)
    };
    decode_f16(bytes)
}

/// Two-stage search over the logical rows in `rows` → `(logical_index,
/// cosine_distance)` ascending, top-`k`. Restricting `rows` (e.g. to a time
/// window from `db::time_range`) narrows the scan itself, so every candidate
/// is in range — filtering a global top-k afterwards can leave nothing.
pub fn search(
    base: &Base,
    tail: &Tail,
    query: &[f32],
    shortlist: usize,
    k: usize,
    rows: Range<usize>,
) -> Vec<(usize, f32)> {
    let qcode = quantize(query);
    let mut heap = scan_base(base, &qcode, shortlist, rows.clone());
    let tail_rows = rows.start.saturating_sub(base.count)..rows.end.saturating_sub(base.count).min(tail.count);
    for t in tail_rows {
        push_bounded(&mut heap, shortlist, hamming(&qcode, tail.code(t)), base.count + t);
    }

    let qnorm = norm(query).max(1e-12);
    let mut scored: Vec<(usize, f32)> = heap
        .into_iter()
        .map(|(_, idx)| {
            let bytes = if idx < base.count {
                base.f16(idx)
            } else {
                tail.f16(idx - base.count)
            };
            let v = decode_f16(bytes);
            let dot: f32 = query.iter().zip(&v).map(|(a, b)| a * b).sum();
            let dist = 1.0 - dot / (qnorm * norm(&v).max(1e-12));
            (idx, dist)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A base where every row holds the same vector — a topic discussed
    /// everywhere, which is exactly what a post-filtered top-k got wrong.
    fn uniform_store(n: usize) -> (PathBuf, Base, Tail) {
        let dir = std::env::temp_dir().join(format!("rust-search-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let v: Vec<f32> = (0..DIM).map(|i| (i as f32).sin()).collect();
        let code = quantize(&v);
        let row: Vec<u8> = v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect();
        std::fs::write(dir.join("codes.bin"), code.repeat(n)).unwrap();
        std::fs::write(dir.join("rerank_f16.bin"), row.repeat(n)).unwrap();
        let _ = std::fs::remove_file(dir.join("tail_codes.bin"));
        let _ = std::fs::remove_file(dir.join("tail_f16.bin"));
        let base = Base::open(&dir, n).unwrap();
        let tail = Tail::load(&dir, n, n, 0, String::new()).unwrap();
        (dir, base, tail)
    }

    #[test]
    fn search_stays_inside_row_range() {
        let (dir, base, mut tail) = uniform_store(100);
        let v: Vec<f32> = (0..DIM).map(|i| (i as f32).sin()).collect();
        tail.append(&vec![v.clone(); 10], 0, "").unwrap();

        let hits = search(&base, &tail, &v, 20, 5, 40..50);
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|(i, _)| (40..50).contains(i)));

        // A range spanning base and tail, and one only in the tail.
        let hits = search(&base, &tail, &v, 50, 50, 95..105);
        let mut idx: Vec<usize> = hits.iter().map(|h| h.0).collect();
        idx.sort();
        assert_eq!(idx, (95..105).collect::<Vec<_>>());
        assert!(search(&base, &tail, &v, 20, 5, 104..110).iter().all(|(i, _)| (104..110).contains(i)));
        assert!(search(&base, &tail, &v, 20, 5, 30..30).is_empty());

        std::fs::remove_dir_all(dir).unwrap();
    }
}
