/* SPDX-License-Identifier: MIT
 *
 * Cross-backend comparator driver: the "originals" column.
 *
 * Independent re-implementations of the PUBLISHED algorithms (not the
 * UniAccurate sources), so the comparison matrix has a reference column that
 * does not share code with the library under test. Demo-only: this file and its
 * build artifacts live under bench/originals/ (gitignored, never packaged).
 *
 * Ported from the accuratesums reference harness (MIT); the MPFR subprocess
 * oracle was dropped -- accuracy is computed by the Nim aggregator against a
 * single in-process superSum/superDot oracle (see bench/aggregate.nim). This
 * driver emits timing + candidate result bit-patterns only.
 *
 * Algorithms, each implemented straight from its paper / canonical description:
 *   naive               plain left-to-right sum
 *   pairwise            iterative adjacent-pair reduction (depth ~log2 n)
 *   kahan               Kahan 1965 compensated summation
 *   neumaier            Neumaier 1977 improved compensated summation
 *   klein               Klein 2006 double-compensation summation
 *   shewchuk            Shewchuk 1997 expansion sum (Grow_Expansion_Zeroelim)
 *   sum2                Ogita-Rump-Oishi 2005, Algorithm 2.4 (faithful, K=2)
 *   naive_dot           plain left-to-right dot product
 *   dot2                Ogita-Rump-Oishi 2005, Algorithm 3.1 (faithful, K=2)
 *
 * Reads the SAME canonical datasets as every other driver (exported once by
 * bench/export_datasets.nim) so inputs are bit-identical across columns.
 *
 * Output (bench/compare/):
 *   perf_originals.csv   algo,dataset,n,type,time_ns,time_per_elem_ns,bw_gbs
 *   cand_originals.csv   algo,dataset,n,type,bits   (uint64 IEEE bit pattern)
 * For f32 rows the f32 result is widened exactly to f64 and emitted as f64
 * bits; the `type` column tells the aggregator which ulp spacing to use.
 *
 * Build: cc -O3 -ffp-contract=off [-mfma] -o bench/originals/driver_originals \
 *          bench/originals/driver_originals.c -lm
 *   (-ffp-contract=off keeps a*b a rounded multiply so TwoProduct's fma is a
 *    true fused multiply-add, not a compiler contraction of the split.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <errno.h>

#define CANON "bench/canonical"
#define OUTDIR "bench/compare"
#define PERF_SCHEMA "algo,dataset,n,type,time_ns,time_per_elem_ns,bw_gbs"
#define CAND_SCHEMA "algo,dataset,n,type,bits"
#define TARGET_NS 50e6  /* per-cell timing budget in ns (~50 ms), matches driver_nim */

static atomic_uint_fast64_t SINK;

static void consume_d(double v) {
    uint64_t b; memcpy(&b, &v, sizeof b);
    atomic_fetch_add_explicit(&SINK, b, memory_order_relaxed);
}
static void consume_f(float v) {
    uint32_t b; memcpy(&b, &v, sizeof b);
    atomic_fetch_add_explicit(&SINK, (uint64_t)b, memory_order_relaxed);
}

/* -------------------------------------------------------------------------
 * Error-free transformations (Knuth TwoSum, Dekker/IEEE TwoProduct via fma).
 * ------------------------------------------------------------------------- */
static inline void twosum_d(double a, double b, double *s, double *e) {
    *s = a + b;
    double v = *s - a;
    *e = (a - (*s - v)) + (b - v);
}
static inline void twoprod_d(double a, double b, double *p, double *e) {
    *p = a * b;
    *e = fma(a, b, -(*p));
}
static inline void twosum_f(float a, float b, float *s, float *e) {
    *s = a + b;
    float v = *s - a;
    *e = (a - (*s - v)) + (b - v);
}
static inline void twoprod_f(float a, float b, float *p, float *e) {
    *p = a * b;
    *e = fmaf(a, b, -(*p));
}

/* -------------------------------------------------------------------------
 * Shewchuk 1997, Grow_Expansion_Zeroelim (Figure 6). Adds the scalar b to the
 * non-overlapping expansion e (sorted ascending by magnitude) using TwoSum at
 * each step, producing a non-overlapping expansion h (sorted ascending) that is
 * EXACTLY e + b. Returns the length of h (zeros eliminated).
 * ------------------------------------------------------------------------- */
static int grow_zeroelim_d(int elen, const double *e, double b, double *h) {
    double Q = b, sum, hh, v;
    int h_index = 0;
    for (int i = 0; i < elen; i++) {
        sum = Q + e[i];
        v = sum - Q;                       /* Knuth TwoSum error (magnitude-agnostic) */
        hh = (Q - (sum - v)) + (e[i] - v);
        if (hh != 0.0) h[h_index++] = hh;
        Q = sum;
    }
    if (Q != 0.0) h[h_index++] = Q;
    return h_index;
}
static int grow_zeroelim_f(int elen, const float *e, float b, float *h) {
    float Q = b, sum, hh, v;
    int h_index = 0;
    for (int i = 0; i < elen; i++) {
        sum = Q + e[i];
        v = sum - Q;
        hh = (Q - (sum - v)) + (e[i] - v);
        if (hh != 0.0f) h[h_index++] = hh;
        Q = sum;
    }
    if (Q != 0.0f) h[h_index++] = Q;
    return h_index;
}

/* -------------------------------------------------------------------------
 * Summation algorithms (double). shewchuk needs two scratch buffers of length
 * n+1 (the expansion can grow up to n terms in the worst case).
 * ------------------------------------------------------------------------- */
static double naive_sum_d(const double *x, int n, double *scratch) {
    (void)scratch; double s = 0; for (int i = 0; i < n; i++) s += x[i]; return s;
}
static double pairwise_sum_d(const double *x, int n, double *scratch) {
    double *b = scratch;
    memcpy(b, x, (size_t)n * sizeof(double));
    while (n > 1) {
        int m = 0;
        for (int i = 0; i + 1 < n; i += 2) b[m++] = b[i] + b[i + 1];
        if (n & 1) b[m++] = b[n - 1];
        n = m;
    }
    return b[0];
}
static double kahan_sum_d(const double *x, int n, double *scratch) {
    (void)scratch; double s = x[0], c = 0;
    for (int i = 1; i < n; i++) { double y = x[i] - c, t = s + y; c = (t - s) - y; s = t; }
    return s;
}
static double neumaier_sum_d(const double *x, int n, double *scratch) {
    (void)scratch; double s = x[0], c = 0;
    for (int i = 1; i < n; i++) {
        double t = s + x[i];
        if (fabs(s) >= fabs(x[i])) c += (s - t) + x[i];
        else                      c += (x[i] - t) + s;
        s = t;
    }
    return s + c;
}
static double klein_sum_d(const double *x, int n, double *scratch) {
    (void)scratch; double s = 0, c = 0, cc = 0;
    for (int i = 0; i < n; i++) {
        double cs, ccs;
        twosum_d(s, x[i], &s, &cs);
        twosum_d(c, cs, &c, &ccs);
        cc += ccs;
    }
    return s + c + cc;
}
static double shewchuk_sum_d(const double *x, int n, double *scratch) {
    double *A = scratch, *B = scratch + (n + 1);
    int lenA = 1; A[0] = x[0];
    for (int i = 1; i < n; i++) {
        int lenB = grow_zeroelim_d(lenA, A, x[i], B);
        double *t = A; A = B; B = t;
        lenA = lenB;
    }
    /* A is a non-overlapping expansion sorted ascending by magnitude. Reducing
     * it in working precision loses the residual when large components cancel.
     * long double (80-bit, 64-bit mantissa on amd64) holds every component >=
     * 0.5 ulp of the result, so a long-double accumulation followed by a single
     * rounding is a correctly-rounded extraction. */
    long double s = 0;
    for (int i = 0; i < lenA; i++) s += (long double)A[i];
    return (double)s;
}
/* ORO Algorithm 2.4 (sum2, faithful): one TwoSum pass + naive error sum. */
static double sum2_d(const double *x, int n, double *scratch) {
    double s = x[0];
    double *err = scratch;
    for (int i = 1; i < n; i++) {
        double e; twosum_d(s, x[i], &s, &e); err[i - 1] = e;
    }
    double es = 0; for (int i = 0; i < n - 1; i++) es += err[i];
    return s + es;
}

/* float summation algorithms (mirror). */
static float naive_sum_f(const float *x, int n, float *scratch) {
    (void)scratch; float s = 0; for (int i = 0; i < n; i++) s += x[i]; return s;
}
static float pairwise_sum_f(const float *x, int n, float *scratch) {
    float *b = scratch; memcpy(b, x, (size_t)n * sizeof(float));
    while (n > 1) {
        int m = 0;
        for (int i = 0; i + 1 < n; i += 2) b[m++] = b[i] + b[i + 1];
        if (n & 1) b[m++] = b[n - 1];
        n = m;
    }
    return b[0];
}
static float kahan_sum_f(const float *x, int n, float *scratch) {
    (void)scratch; float s = x[0], c = 0;
    for (int i = 1; i < n; i++) { float y = x[i] - c, t = s + y; c = (t - s) - y; s = t; }
    return s;
}
static float neumaier_sum_f(const float *x, int n, float *scratch) {
    (void)scratch; float s = x[0], c = 0;
    for (int i = 1; i < n; i++) {
        float t = s + x[i];
        if (fabsf(s) >= fabsf(x[i])) c += (s - t) + x[i];
        else                        c += (x[i] - t) + s;
        s = t;
    }
    return s + c;
}
static float klein_sum_f(const float *x, int n, float *scratch) {
    (void)scratch; float s = 0, c = 0, cc = 0;
    for (int i = 0; i < n; i++) {
        float cs, ccs;
        twosum_f(s, x[i], &s, &cs);
        twosum_f(c, cs, &c, &ccs);
        cc += ccs;
    }
    return s + c + cc;
}
static float shewchuk_sum_f(const float *x, int n, float *scratch) {
    float *A = scratch, *B = scratch + (n + 1);
    int lenA = 1; A[0] = x[0];
    for (int i = 1; i < n; i++) {
        int lenB = grow_zeroelim_f(lenA, A, x[i], B);
        float *t = A; A = B; B = t;
        lenA = lenB;
    }
    long double s = 0;
    for (int i = 0; i < lenA; i++) s += (long double)A[i];
    return (float)s;
}
static float sum2_f(const float *x, int n, float *scratch) {
    float s = x[0];
    float *err = scratch;
    for (int i = 1; i < n; i++) { float e; twosum_f(s, x[i], &s, &e); err[i - 1] = e; }
    float es = 0; for (int i = 0; i < n - 1; i++) es += err[i];
    return s + es;
}

/* Dot products. */
static double naive_dot_d(const double *x, const double *y, int n, double *scratch) {
    (void)scratch; double s = 0; for (int i = 0; i < n; i++) s += x[i] * y[i]; return s;
}
/* ORO Algorithm 3.1 (dot2, faithful). */
static double dot2_d(const double *x, const double *y, int n, double *scratch) {
    (void)scratch; double p, s;
    twoprod_d(x[0], y[0], &p, &s);
    for (int i = 1; i < n; i++) {
        double pi, ei, t;
        twoprod_d(x[i], y[i], &pi, &ei);
        twosum_d(p, pi, &p, &t);
        s += t + ei;
    }
    return p + s;
}
static float naive_dot_f(const float *x, const float *y, int n, float *scratch) {
    (void)scratch; float s = 0; for (int i = 0; i < n; i++) s += x[i] * y[i]; return s;
}
static float dot2_f(const float *x, const float *y, int n, float *scratch) {
    (void)scratch; float p, s;
    twoprod_f(x[0], y[0], &p, &s);
    for (int i = 1; i < n; i++) {
        float pi, ei, t;
        twoprod_f(x[i], y[i], &pi, &ei);
        twosum_f(p, pi, &p, &t);
        s += t + ei;
    }
    return p + s;
}

/* -------------------------------------------------------------------------
 * Adaptive timing (Google-Benchmark style): ramp a probe to ~5 ms, estimate the
 * per-call cost, then size the measured run to ~TARGET_NS (capped). The atomic
 * sink prevents dead-code elimination of the result. GCC statement expressions
 * let the macro time any expression in scope.
 * ------------------------------------------------------------------------- */
#define TIME_NS_D(call_expr, out_val) ({                                        \
    double _r;                                                                  \
    for (int _w = 0; _w < 10; _w++) { _r = (call_expr); consume_d(_r); }        \
    long _m = 1; double _dt; struct timespec _a, _b;                            \
    do {                                                                        \
        clock_gettime(CLOCK_MONOTONIC, &_a);                                    \
        for (long _i = 0; _i < _m; _i++) { _r = (call_expr); consume_d(_r); }   \
        clock_gettime(CLOCK_MONOTONIC, &_b);                                    \
        _dt = (_b.tv_sec - _a.tv_sec) + (_b.tv_nsec - _a.tv_nsec) / 1e9;        \
        if (_dt >= 5e-3 || _m >= 1000) break;                                   \
        _m *= 10;                                                               \
    } while (1);                                                                \
    double _per = _dt / _m; if (_per <= 0) _per = 1e-9;                         \
    long _it = (long)(TARGET_NS / (_per * 1e9));                                \
    if (_it < 1) _it = 1; if (_it > (1L << 22)) _it = (1L << 22);              \
    clock_gettime(CLOCK_MONOTONIC, &_a);                                        \
    for (long _i = 0; _i < _it; _i++) { _r = (call_expr); consume_d(_r); }      \
    clock_gettime(CLOCK_MONOTONIC, &_b);                                        \
    double _ns = ((_b.tv_sec - _a.tv_sec) + (_b.tv_nsec - _a.tv_nsec)/1e9)      \
                 / _it * 1e9;                                                   \
    (out_val) = _r; _ns; })

#define TIME_NS_F(call_expr, out_val) ({                                        \
    float _r;                                                                   \
    for (int _w = 0; _w < 10; _w++) { _r = (call_expr); consume_f(_r); }        \
    long _m = 1; double _dt; struct timespec _a, _b;                            \
    do {                                                                        \
        clock_gettime(CLOCK_MONOTONIC, &_a);                                    \
        for (long _i = 0; _i < _m; _i++) { _r = (call_expr); consume_f(_r); }   \
        clock_gettime(CLOCK_MONOTONIC, &_b);                                    \
        _dt = (_b.tv_sec - _a.tv_sec) + (_b.tv_nsec - _a.tv_nsec) / 1e9;        \
        if (_dt >= 5e-3 || _m >= 1000) break;                                   \
        _m *= 10;                                                               \
    } while (1);                                                                \
    double _per = _dt / _m; if (_per <= 0) _per = 1e-9;                         \
    long _it = (long)(TARGET_NS / (_per * 1e9));                                \
    if (_it < 1) _it = 1; if (_it > (1L << 22)) _it = (1L << 22);              \
    clock_gettime(CLOCK_MONOTONIC, &_a);                                        \
    for (long _i = 0; _i < _it; _i++) { _r = (call_expr); consume_f(_r); }      \
    clock_gettime(CLOCK_MONOTONIC, &_b);                                        \
    double _ns = ((_b.tv_sec - _a.tv_sec) + (_b.tv_nsec - _a.tv_nsec)/1e9)      \
                 / _it * 1e9;                                                   \
    (out_val) = _r; _ns; })

/* Emit a candidate result as its uint64 IEEE bit pattern (f32 widened to f64). */
static uint64_t bits_d(double v) { uint64_t b; memcpy(&b, &v, sizeof b); return b; }

static void emit_perf(FILE *f, const char *algo, const char *ds, int n, const char *typ,
                      double tns, double per_elem, double bw) {
    fprintf(f, "%s,%s,%d,%s,%.3f,%.4f,%.2f\n", algo, ds, n, typ, tns, per_elem, bw);
}
static void emit_cand(FILE *f, const char *algo, const char *ds, int n, const char *typ,
                      uint64_t bits) {
    fprintf(f, "%s,%s,%d,%s,%llu\n", algo, ds, n, typ, (unsigned long long)bits);
}

/* -------------------------------------------------------------------------
 * Canonical dataset loading (manifest.csv + little-endian raw f64/f32 .bin).
 * ------------------------------------------------------------------------- */
typedef struct { char typ[8], dataset[32]; int n, bpe; char x[128], y[128]; } Entry;

static int load_manifest(Entry *ents, int max) {
    char path[256]; snprintf(path, sizeof path, "%s/manifest.csv", CANON);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "open %s: ", path); perror(""); exit(1); }
    char line[1024]; int n = 0;
    if (!fgets(line, sizeof line, f)) { fclose(f); return 0; }  /* header */
    while (fgets(line, sizeof line, f) && n < max) {
        Entry *e = &ents[n];
        if (sscanf(line, "%7[^,],%31[^,],%d,%d,%*d,%127[^,],%127s",
                   e->typ, e->dataset, &e->n, &e->bpe, e->x, e->y) == 6)
            n++;
    }
    fclose(f);
    return n;
}

static double *load_f64(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    double *a = malloc((size_t)n * sizeof(double));
    if (fread(a, sizeof(double), (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); return a;
}
static float *load_f32(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    float *a = malloc((size_t)n * sizeof(float));
    if (fread(a, sizeof(float), (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); return a;
}

/* ------------------------------------------------------------------------- */
int main(void) {
    atomic_init(&SINK, 0);
    Entry ents[256];
    int ne = load_manifest(ents, 256);
    if (ne == 0) { fprintf(stderr, "no canonical datasets -- run `nimble benchData` first\n"); return 1; }

    if (access(OUTDIR, F_OK) != 0) { if (mkdir(OUTDIR, 0777) != 0 && errno != EEXIST) { perror("mkdir OUTDIR"); } }

    char p[256];
    snprintf(p, sizeof p, "%s/perf_originals.csv", OUTDIR); FILE *fp = fopen(p, "w");
    snprintf(p, sizeof p, "%s/cand_originals.csv", OUTDIR); FILE *fc = fopen(p, "w");
    if (!fp || !fc) { perror("open output csv"); return 1; }
    fprintf(fp, "%s\n", PERF_SCHEMA); fprintf(fc, "%s\n", CAND_SCHEMA);

    for (int ei = 0; ei < ne; ei++) {
        Entry *e = &ents[ei];
        int as_f32 = strcmp(e->typ, "f32") == 0;
        int n = e->n;
        double *scratch = malloc((size_t)(n + 1) * 2 * sizeof(double));   /* shewchuk A,B */

        if (as_f32) {
            float *x = load_f32(e->x, n), *y = load_f32(e->y, n);
            float *fsc = malloc((size_t)(n + 1) * 2 * sizeof(float));

            struct { const char *name; float (*fn)(const float *, int, float *); } sums[] = {
                {"naive", naive_sum_f}, {"pairwise", pairwise_sum_f},
                {"kahan", kahan_sum_f}, {"neumaier", neumaier_sum_f},
                {"klein", klein_sum_f}, {"shewchuk", shewchuk_sum_f}, {"sum2", sum2_f},
            };
            for (size_t k = 0; k < sizeof sums / sizeof sums[0]; k++) {
                float val; double ns = TIME_NS_F(sums[k].fn(x, n, fsc), val);
                emit_perf(fp, sums[k].name, e->dataset, n, e->typ, ns, ns / n,
                          (e->bpe * n) / ns);
                emit_cand(fc, sums[k].name, e->dataset, n, e->typ, bits_d((double)val));
            }
            struct { const char *name; float (*fn)(const float *, const float *, int, float *); } dots[] = {
                {"naive_dot", naive_dot_f}, {"dot2", dot2_f},
            };
            for (size_t k = 0; k < sizeof dots / sizeof dots[0]; k++) {
                float val; double ns = TIME_NS_F(dots[k].fn(x, y, n, fsc), val);
                emit_perf(fp, dots[k].name, e->dataset, n, e->typ, ns, ns / n,
                          (e->bpe * n) / ns);
                emit_cand(fc, dots[k].name, e->dataset, n, e->typ, bits_d((double)val));
            }
            free(fsc); free(x); free(y);
        } else {
            double *x = load_f64(e->x, n), *y = load_f64(e->y, n);
            struct { const char *name; double (*fn)(const double *, int, double *); } sums[] = {
                {"naive", naive_sum_d}, {"pairwise", pairwise_sum_d},
                {"kahan", kahan_sum_d}, {"neumaier", neumaier_sum_d},
                {"klein", klein_sum_d}, {"shewchuk", shewchuk_sum_d}, {"sum2", sum2_d},
            };
            for (size_t k = 0; k < sizeof sums / sizeof sums[0]; k++) {
                double val; double ns = TIME_NS_D(sums[k].fn(x, n, scratch), val);
                emit_perf(fp, sums[k].name, e->dataset, n, e->typ, ns, ns / n,
                          (e->bpe * n) / ns);
                emit_cand(fc, sums[k].name, e->dataset, n, e->typ, bits_d(val));
            }
            struct { const char *name; double (*fn)(const double *, const double *, int, double *); } dots[] = {
                {"naive_dot", naive_dot_d}, {"dot2", dot2_d},
            };
            for (size_t k = 0; k < sizeof dots / sizeof dots[0]; k++) {
                double val; double ns = TIME_NS_D(dots[k].fn(x, y, n, scratch), val);
                emit_perf(fp, dots[k].name, e->dataset, n, e->typ, ns, ns / n,
                          (e->bpe * n) / ns);
                emit_cand(fc, dots[k].name, e->dataset, n, e->typ, bits_d(val));
            }
            free(x); free(y);
        }
        free(scratch);
        fprintf(stderr, "[originals] %s,%s,%d\n", e->typ, e->dataset, n);
    }
    fclose(fp); fclose(fc);
    fprintf(stderr, "driver_originals done. sink=%llu\n",
            (unsigned long long)atomic_load(&SINK));
    return 0;
}
