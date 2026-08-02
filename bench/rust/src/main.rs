// SPDX-License-Identifier: MIT
// Cross-backend comparator driver: the Rust backend (`accurate` crate v0.4).
// Reads the SAME canonical datasets as every other driver (exported by
// bench/export_datasets.nim) so inputs are bit-identical across columns.
//
// Implements (algo-vs-algo rule: a cell only for the lines the crate exposes):
//   sums: naive (NaiveSum), kahan (Kahan), neumaier (Neumaier), klein (Klein),
//         sum2 (Sum2), sumK (Sum3, K=3)
//   dots: naive_dot (NaiveDot), dot2 (Dot2), dotK (Dot3, K=3)
// The crate has no pairwise/acc/near/shewchuk surface -- that subset is
// intentionally best-effort, not silently skipped.
//
// Ported from the accuratesums reference harness (MIT); the MPFR subprocess
// oracle was dropped -- accuracy is computed by the Nim aggregator against a
// single in-process superSum/superDot oracle. This driver emits timing +
// candidate result bit-patterns only.
//
// Output (bench/compare/):
//   perf_rust_raw.csv   algo,dataset,n,type,time_ns,time_per_elem_ns,bw_gbs
//   cand_rust_raw.csv   algo,dataset,n,type,bits   (u64 IEEE bit pattern; f32
//                       widened exactly to f64, `type` selects ulp spacing)
// Disposable intermediate names: the nimble tasks rename them to the tagged
// perf_rust.csv / perf_rust_fma.csv the aggregator expects, so the FMA variant
// never clobbers the base column (mirrors the C driver's perf_originals.csv).
// Gitignored. Build/run: `cargo run --release [--no-default-features]` from
// bench/rust (see nimble tasks benchRust / benchRustFma).

use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use accurate::traits::*;
use accurate::sum::{Kahan, Klein, NaiveSum, Neumaier, Sum2, Sum3};
use accurate::dot::{Dot2, Dot3, NaiveDot};

const CANON: &str = "bench/canonical";
const OUTDIR: &str = "bench/compare";
const PERF_SCHEMA: &str = "algo,dataset,n,type,time_ns,time_per_elem_ns,bw_gbs";
const CAND_SCHEMA: &str = "algo,dataset,n,type,bits";
const TARGET_NS: f64 = 50e6;

static SINK: AtomicU64 = AtomicU64::new(0);

fn consume_f64(v: f64) {
    std::hint::black_box(v);
    SINK.fetch_add(v.to_bits(), Ordering::Relaxed);
}
fn consume_f32(v: f32) {
    std::hint::black_box(v);
    SINK.fetch_add(v.to_bits() as u64, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Canonical dataset loading (little-endian raw f64/f32 .bin + manifest.csv).
// ---------------------------------------------------------------------------
struct Entry {
    typ: String,
    dataset: String,
    n: usize,
    bpe: usize,
    x: String,
    y: String,
}

fn load_manifest() -> Vec<Entry> {
    let f = File::open(format!("{}/manifest.csv", CANON)).expect("manifest");
    let mut rows = Vec::new();
    for line in BufReader::new(f).lines().skip(1).flatten() {
        let p: Vec<&str> = line.split(',').collect();
        if p.len() < 7 {
            continue;
        }
        rows.push(Entry {
            typ: p[0].into(),
            dataset: p[1].into(),
            n: p[2].parse().unwrap(),
            bpe: p[3].parse().unwrap(),
            x: p[5].into(),
            y: p[6].into(),
        });
    }
    rows
}

fn load_f64(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).expect("bin");
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}
fn load_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).expect("bin");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Adaptive timing: ramp a probe to ~5 ms, estimate per-call cost, size the
// measured run to ~TARGET_NS (capped). black_box + atomic sink prevent DCE.
// ---------------------------------------------------------------------------
fn time_fn<F: FnMut() -> f64>(mut f: F) -> f64 {
    for _ in 0..10 {
        consume_f64(f());
    }
    let mut m: u64 = 1;
    let mut dt;
    loop {
        let t0 = Instant::now();
        for _ in 0..m {
            consume_f64(f());
        }
        dt = t0.elapsed().as_secs_f64();
        if dt >= 5e-3 || m >= 1000 {
            break;
        }
        m *= 10;
    }
    let per = dt / m as f64;
    let iters = ((TARGET_NS / (per * 1e9)).max(1.0).min((1 << 22) as f64)) as u64;
    let t0 = Instant::now();
    for _ in 0..iters {
        consume_f64(f());
    }
    t0.elapsed().as_secs_f64() / iters as f64 * 1e9
}
fn time_fn_f32<F: FnMut() -> f32>(mut f: F) -> f64 {
    for _ in 0..10 {
        consume_f32(f());
    }
    let mut m: u64 = 1;
    let mut dt;
    loop {
        let t0 = Instant::now();
        for _ in 0..m {
            consume_f32(f());
        }
        dt = t0.elapsed().as_secs_f64();
        if dt >= 5e-3 || m >= 1000 {
            break;
        }
        m *= 10;
    }
    let per = dt / m as f64;
    let iters = ((TARGET_NS / (per * 1e9)).max(1.0).min((1 << 22) as f64)) as u64;
    let t0 = Instant::now();
    for _ in 0..iters {
        consume_f32(f());
    }
    t0.elapsed().as_secs_f64() / iters as f64 * 1e9
}

fn write_perf(f: &mut File, algo: &str, ds: &str, n: usize, typ: &str, tns: f64, bw: f64) {
    let _ = writeln!(f, "{},{},{},{},{:.3},{:.4},{:.2}", algo, ds, n, typ, tns, tns / n as f64, bw);
}
fn write_cand(f: &mut File, algo: &str, ds: &str, n: usize, typ: &str, bits: u64) {
    let _ = writeln!(f, "{},{},{},{},{}", algo, ds, n, typ, bits);
}

// ---------------------------------------------------------------------------
// Per-algo measurement: generic over the accumulator type. Writes the perf row
// and the candidate bit-pattern row.
// ---------------------------------------------------------------------------
fn run_sum_f64<A: SumAccumulator<f64>>(algo: &str, x: &[f64], f_perf: &mut File, f_cand: &mut File, e: &Entry) {
    let ns = time_fn(|| x.iter().copied().sum_with_accumulator::<A>());
    let val = x.iter().copied().sum_with_accumulator::<A>();
    let bw = (e.bpe as f64 * e.n as f64) / ns;
    write_perf(f_perf, algo, &e.dataset, e.n, &e.typ, ns, bw);
    write_cand(f_cand, algo, &e.dataset, e.n, &e.typ, val.to_bits());
}
fn run_sum_f32<A: SumAccumulator<f32>>(algo: &str, x: &[f32], f_perf: &mut File, f_cand: &mut File, e: &Entry) {
    let ns = time_fn_f32(|| x.iter().copied().sum_with_accumulator::<A>());
    let val: f32 = x.iter().copied().sum_with_accumulator::<A>();
    let bw = (e.bpe as f64 * e.n as f64) / ns;
    write_perf(f_perf, algo, &e.dataset, e.n, &e.typ, ns, bw);
    // f32 widened exactly to f64, emitted as f64 bits (type column = f32).
    write_cand(f_cand, algo, &e.dataset, e.n, &e.typ, (val as f64).to_bits());
}
fn run_dot_f64<A: DotAccumulator<f64>>(algo: &str, x: &[f64], y: &[f64], f_perf: &mut File, f_cand: &mut File, e: &Entry) {
    let ns = time_fn(|| x.iter().copied().zip(y.iter().copied()).dot_with_accumulator::<A>());
    let val = x.iter().copied().zip(y.iter().copied()).dot_with_accumulator::<A>();
    let bw = (e.bpe as f64 * e.n as f64) / ns;
    write_perf(f_perf, algo, &e.dataset, e.n, &e.typ, ns, bw);
    write_cand(f_cand, algo, &e.dataset, e.n, &e.typ, val.to_bits());
}
fn run_dot_f32<A: DotAccumulator<f32>>(algo: &str, x: &[f32], y: &[f32], f_perf: &mut File, f_cand: &mut File, e: &Entry) {
    let ns = time_fn_f32(|| x.iter().copied().zip(y.iter().copied()).dot_with_accumulator::<A>());
    let val: f32 = x.iter().copied().zip(y.iter().copied()).dot_with_accumulator::<A>();
    let bw = (e.bpe as f64 * e.n as f64) / ns;
    write_perf(f_perf, algo, &e.dataset, e.n, &e.typ, ns, bw);
    write_cand(f_cand, algo, &e.dataset, e.n, &e.typ, (val as f64).to_bits());
}

fn run_entry(e: &Entry, f_perf: &mut File, f_cand: &mut File) {
    let as_f32 = e.typ == "f32";
    if as_f32 {
        let x = load_f32(&e.x);
        let y = load_f32(&e.y);
        assert_eq!(x.len(), e.n, "canonical size mismatch");
        run_sum_f32::<NaiveSum<f32>>("naive", &x, f_perf, f_cand, e);
        run_sum_f32::<Kahan<f32>>("kahan", &x, f_perf, f_cand, e);
        run_sum_f32::<Neumaier<f32>>("neumaier", &x, f_perf, f_cand, e);
        run_sum_f32::<Klein<f32>>("klein", &x, f_perf, f_cand, e);
        run_sum_f32::<Sum2<f32>>("sum2", &x, f_perf, f_cand, e);
        run_sum_f32::<Sum3<f32>>("sumK", &x, f_perf, f_cand, e);
        run_dot_f32::<NaiveDot<f32>>("naive_dot", &x, &y, f_perf, f_cand, e);
        run_dot_f32::<Dot2<f32>>("dot2", &x, &y, f_perf, f_cand, e);
        run_dot_f32::<Dot3<f32>>("dotK", &x, &y, f_perf, f_cand, e);
    } else {
        let x = load_f64(&e.x);
        let y = load_f64(&e.y);
        assert_eq!(x.len(), e.n, "canonical size mismatch");
        run_sum_f64::<NaiveSum<f64>>("naive", &x, f_perf, f_cand, e);
        run_sum_f64::<Kahan<f64>>("kahan", &x, f_perf, f_cand, e);
        run_sum_f64::<Neumaier<f64>>("neumaier", &x, f_perf, f_cand, e);
        run_sum_f64::<Klein<f64>>("klein", &x, f_perf, f_cand, e);
        run_sum_f64::<Sum2<f64>>("sum2", &x, f_perf, f_cand, e);
        run_sum_f64::<Sum3<f64>>("sumK", &x, f_perf, f_cand, e);
        run_dot_f64::<NaiveDot<f64>>("naive_dot", &x, &y, f_perf, f_cand, e);
        run_dot_f64::<Dot2<f64>>("dot2", &x, &y, f_perf, f_cand, e);
        run_dot_f64::<Dot3<f64>>("dotK", &x, &y, f_perf, f_cand, e);
    }
    eprintln!("[rust] {},{},{}", e.typ, e.dataset, e.n);
}

fn main() {
    create_dir_all(OUTDIR).unwrap();
    let entries = load_manifest();
    if entries.is_empty() {
        panic!("no canonical datasets -- run `nimble benchData` first");
    }
    let mut f_perf = File::create(format!("{}/perf_rust_raw.csv", OUTDIR)).unwrap();
    let mut f_cand = File::create(format!("{}/cand_rust_raw.csv", OUTDIR)).unwrap();
    let _ = writeln!(f_perf, "{}", PERF_SCHEMA);
    let _ = writeln!(f_cand, "{}", CAND_SCHEMA);
    for e in &entries {
        run_entry(e, &mut f_perf, &mut f_cand);
    }
    eprintln!("driver_rust done. sink={}", SINK.load(Ordering::Relaxed));
}
