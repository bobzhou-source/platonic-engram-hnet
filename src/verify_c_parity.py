"""Bit-parity gate: C (device-style, integer-only) vs numpy vs pure-Python vs torch QAT.

    python src/verify_c_parity.py [--models-dir weights] [--ckpt runs/.../cspw_full.pt] [--fuzz 500]

Two engines, each checked against an independent C reference compiled with
``cc -O2 -std=c11 -ffp-contract=off``:

* ConceptEngine (PLHN + ENGR, unigram + Bigram-Span Engram): golden texts, mixed-script corpus lines,
  random UTF-8 and malformed byte strings -> int8[128] must match numpy / pure-Python / torch QAT.
* CS-PW (``cspw.bin`` + an SGRF story graph, default ``weights/booksum-story-graph.idx``): Chunk-GQA hybrid, Q15 Chrono-CfC decay, frame/phrase pointer
  selection on random scene sets (deployed and random weights), plus the full device lookup (alias reveal
  gate, spoiler gate, scene selection, frame fill) on the deployed and on random story graphs -> slots,
  g, tau, pointer indices and refresher bytes must match ``cspw_model``.

Also validates PLHN/ENGR/ICPT/CSPW/SGRF byte layouts and the goldens (golden.jsonl, golden_vectors.bin,
cspw_golden.bin).  Exit status 0 only when everything matches.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cspw_model as CM  # noqa: E402
import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "weights"

C_SOURCE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DB 64
#define DC 128
#define DR 32
#define MAXB 256
#define MAXSPAN 12
#define EDIMS 24
#define BIGRAM_MAX 18

typedef struct {
  int se, sr, sp, sc;
  const int8_t *emb, *c1dw, *c1pw, *c1g, *c2dw, *c2pw, *c2g, *rq, *rk, *cp, *cdw, *cpw, *cg, *co;
} Plhn;

static int32_t shr(int32_t a, int s) { return s > 0 ? ((a + (1 << (s - 1))) >> s) : (s < 0 ? a * (1 << -s) : a); }
static int32_t sat8(int32_t a) { return a < -127 ? -127 : (a > 127 ? 127 : a); }
static int64_t sat8l(int64_t a) { return a < -127 ? -127 : (a > 127 ? 127 : a); }
static int32_t rq(int32_t a, int s) { return sat8(shr(a, s)); }
static int shift_of(float scale) { return (int)lroundf(-log2f(scale)); }
static uint64_t fnv(const uint8_t *p, int n) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
  return h;
}
static int u8len(uint8_t b) { return b < 0x80 ? 1 : b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1; }

/* y[j] = sum_i x[i] * W[i*out + j] */
static void matvec(const int32_t *x, const int8_t *W, int in, int out, int s, int32_t *y) {
  for (int j = 0; j < out; j++) {
    int32_t acc = 0;
    for (int i = 0; i < in; i++) acc += x[i] * (int32_t)W[i * out + j];
    y[j] = rq(acc, s);
  }
}

/* gated ShortConv block over T rows of width d (in place). out==NULL -> encoder block */
static void block(int32_t *x, int T, int d, const int8_t *gate, const int8_t *dw, const int8_t *pw,
                  const int8_t *out, int s) {
  int32_t *gs = malloc(sizeof(int32_t) * T * d), *y = malloc(sizeof(int32_t) * T * d);
  int32_t z[DC], o[DC], r[DC];
  for (int t = 0; t < T; t++) {
    matvec(x + t * d, gate, d, d, s, gs + t * d);
    for (int c = 0; c < d; c++) {
      int32_t g = gs[t * d + c] + 64;
      g = g < 0 ? 0 : (g > 128 ? 128 : g);
      gs[t * d + c] = g;
      y[t * d + c] = (x[t * d + c] * g + 64) >> 7;
    }
  }
  for (int t = 0; t < T; t++) {
    for (int c = 0; c < d; c++) {
      int32_t a = dw[c * 3 + 2] * y[t * d + c];
      if (t >= 1) a += dw[c * 3 + 1] * y[(t - 1) * d + c];
      if (t >= 2) a += dw[c * 3 + 0] * y[(t - 2) * d + c];
      z[c] = rq(a, s);
    }
    matvec(z, pw, d, d, s, o);
    for (int c = 0; c < d; c++) o[c] = (o[c] * gs[t * d + c] + 64) >> 7;
    if (out) matvec(o, out, d, d, s, r); else memcpy(r, o, sizeof(int32_t) * d);
    for (int c = 0; c < d; c++) x[t * d + c] = sat8(x[t * d + c] + r[c]);
  }
  free(gs); free(y);
}

static uint64_t isqrt64(uint64_t n) {
  uint64_t r = (uint64_t)sqrt((double)n);
  while (r * r > n) r--;
  while ((r + 1) * (r + 1) <= n) r++;
  return r;
}

static void encode(const Plhn *m, const uint8_t *eng, uint32_t eslots, int esh, int bigram, const uint8_t *b,
                   int n, int8_t out[DC]) {
  if (n > MAXB) n = MAXB;
  memset(out, 0, DC);
  if (n == 0) return;
  static int32_t h[MAXB * DB], q[MAXB * DR], k[MAXB * DR], xs[MAXB * DC];
  for (int t = 0; t < n; t++)
    for (int d = 0; d < DB; d++) h[t * DB + d] = sat8(shr(m->emb[b[t] * DB + d], m->se - 5));
  block(h, n, DB, m->c1g, m->c1dw, m->c1pw, NULL, m->sp);
  block(h, n, DB, m->c2g, m->c2dw, m->c2pw, NULL, m->sp);
  for (int t = 0; t < n; t++) {
    matvec(h + t * DB, m->rq, DB, DR, m->sr, q + t * DR);
    matvec(h + t * DB, m->rk, DB, DR, m->sr, k + t * DR);
  }
  static const int32_t RECIP[MAXSPAN + 1] = {0, 65536, 32768, 21845, 16384, 13107, 10923, 9362, 8192, 7282, 6554, 5958, 5461};
  int nc = 0, start = 0, len = 0, prev_start = 0;
  for (int t = 0; t <= n; t++) {
    int boundary;
    if (t == n) boundary = 1;
    else if (t == 0 || len >= MAXSPAN) boundary = 1;
    else if ((b[t] & 0xC0) == 0x80) boundary = 0;
    else {
      int32_t dot = 0;
      for (int j = 0; j < DR; j++) dot += q[t * DR + j] * k[(t - 1) * DR + j];
      boundary = (len + u8len(b[t]) > MAXSPAN) || dot <= 0;
    }
    if (boundary && t > 0) {
      int L = t - start;
      int32_t mv[DB];
      for (int d = 0; d < DB; d++) {
        int32_t sum = 0;
        for (int u = start; u < t; u++) sum += h[u * DB + d];
        mv[d] = sat8((sum * RECIP[L] + 32768) >> 16);
      }
      int32_t *x = xs + nc * DC;
      matvec(mv, m->cp, DB, DC, m->sp, x);
      if (eng) {
        uint32_t slot = (uint32_t)(fnv(b + start, L) % eslots);
        const int8_t *delta = (const int8_t *)(eng + 512 + (size_t)slot * 64 + 8);
        for (int d = 0; d < EDIMS; d++) x[d] = sat8(x[d] + sat8(shr(delta[d], esh - 5)));
        if (bigram && nc >= 1 && t - prev_start <= BIGRAM_MAX) {
          uint64_t h2 = fnv(b + prev_start, t - prev_start), stored;
          uint32_t s2 = (uint32_t)(h2 % eslots);
          memcpy(&stored, eng + 512 + (size_t)s2 * 64, 8);
          if (h2 != 0 && stored == h2) {
            const int8_t *bd = (const int8_t *)(eng + 512 + (size_t)s2 * 64 + 8);
            for (int d = 0; d < EDIMS; d++) x[d] = sat8(x[d] + sat8(shr(bd[d] >> 1, esh - 5)));
          }
        }
      }
      prev_start = start;
      nc++;
    }
    if (boundary) { start = t; len = 0; }
    len++;
  }
  for (int loop = 0; loop < 2; loop++) block(xs, nc, DC, m->cg, m->cdw, m->cpw, m->co, m->sc);
  int32_t s[DC];
  uint64_t n2 = 0;
  for (int d = 0; d < DC; d++) {
    s[d] = 0;
    for (int c = 0; c < nc; c++) s[d] += xs[c * DC + d];
    n2 += (uint64_t)((int64_t)s[d] * s[d]);
  }
  int64_t nn = (int64_t)isqrt64(n2);
  if (nn == 0) return;
  for (int d = 0; d < DC; d++) {
    int64_t a = 127 * (int64_t)s[d];
    int64_t v = a >= 0 ? (a + nn / 2) / nn : -((-a + nn / 2) / nn);
    out[d] = (int8_t)sat8((int32_t)v);
  }
}

/* ------------------------------------------------------------------ CS-PW */
#define DM 192
#define DS 128
#define DKV 96
#define HD 48
#define CSPW_BYTES 393728

typedef struct {
  const int8_t *up, *wq, *wk, *wv, *wo, *down, *sw, *gate, *fb, *pb;
  int sb;
  const uint8_t *fs, *ps;
} Cspw;

static const int64_t EXPL[65] = {256, 235, 215, 197, 181, 166, 152, 140, 128, 117, 108, 99, 91, 83, 76, 70, 64, 59,
  54, 49, 45, 41, 38, 35, 32, 29, 27, 25, 23, 21, 19, 17, 16, 15, 13, 12, 11, 10, 10, 9, 8, 7, 7, 6, 6, 5, 5, 4,
  4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0};
static const int64_t FRAC[16] = {32768, 34219, 35734, 37316, 38968, 40693, 42495, 44376, 46341, 48393, 50535,
  52773, 55109, 57549, 60097, 62757};
static const int64_t NEGQ[17] = {32768, 31379, 30048, 28774, 27554, 26386, 25268, 24196, 23170, 22188, 21247,
  20347, 19484, 18658, 17867, 17109, 16384};

typedef struct {
  int g[4], tau[4];
  int64_t slots[4][DS];
  int64_t u[4][4][DS];
  int frame[4], phrase, arc;
} WeaveOut;

static int64_t fdiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b) != 0 && ((a < 0) != (b < 0))) q--;
  return q;
}
static int tau_of(int g) {
  int code = 64 + (g >> 1);
  int64_t t = (FRAC[code & 15] * ((int64_t)1 << (code >> 4)) + 16384) >> 15;
  return t < 16 ? 16 : (int)t;
}
static int64_t decay_q15(int64_t dp, int64_t tau) {
  int64_t x = (dp * 256) / tau;
  if (x >= 4096) return 0;
  int ip = (int)(x >> 8), xf = (int)(x & 255), seg = xf >> 4, t = xf & 15;
  int64_t base = (NEGQ[seg] * (16 - t) + NEGQ[seg + 1] * t + 8) >> 4;
  return ip == 0 ? base : (base + ((int64_t)1 << (ip - 1))) >> ip;
}

static int parse_cspw(const uint8_t *b, long n, Cspw *w) {
  if (n != CSPW_BYTES || memcmp(b, "CSPW", 4) != 0) return 0;
  uint16_t ver, flags; uint32_t dims[6];
  memcpy(&ver, b + 4, 2); memcpy(&flags, b + 6, 2); memcpy(dims, b + 8, 24);
  if (ver != 2 || flags != 0 || dims[0] != DM || dims[1] != DS || dims[2] != 4 || dims[3] != 256 || dims[4] != 512)
    return 0;
  w->up = (const int8_t *)(b + 512); w->wq = (const int8_t *)(b + 25088); w->wk = (const int8_t *)(b + 61952);
  w->wv = (const int8_t *)(b + 80384); w->wo = (const int8_t *)(b + 98816); w->down = (const int8_t *)(b + 135680);
  w->sw = (const int8_t *)(b + 160256);
  int16_t sb; memcpy(&sb, b + 160384, 2); w->sb = sb;
  w->gate = (const int8_t *)(b + 160512); w->fb = (const int8_t *)(b + 226048);
  w->fs = b + 258816; w->pb = (const int8_t *)(b + 275200); w->ps = b + 340736;
  return 1;
}

static void weave(const Cspw *w, int M, const int8_t S[4][DS], const uint32_t *p, const int8_t *echo, WeaveOut *o) {
  int64_t h[4][DM], q[4][DM], k[4][DKV], v[4][DKV], at[4][DM], r[4][DS];
  memset(o, 0, sizeof *o);
  for (int m = 0; m < M; m++)
    for (int j = 0; j < DM; j++) {
      int64_t a = 0;
      for (int i = 0; i < DS; i++) a += (int64_t)S[m][i] * w->up[i * DM + j];
      h[m][j] = sat8l(a >> 7);
    }
  for (int m = 0; m < M; m++) {
    for (int j = 0; j < DM; j++) {
      int64_t a = 0;
      for (int i = 0; i < DM; i++) a += h[m][i] * w->wq[i * DM + j];
      q[m][j] = sat8l(a >> 7);
    }
    for (int j = 0; j < DKV; j++) {
      int64_t a = 0, c = 0;
      for (int i = 0; i < DM; i++) { a += h[m][i] * w->wk[i * DKV + j]; c += h[m][i] * w->wv[i * DKV + j]; }
      k[m][j] = sat8l(a >> 7); v[m][j] = sat8l(c >> 7);
    }
  }
  for (int hd = 0; hd < 4; hd++) {
    int kv = hd >> 1;
    for (int a = 0; a < M; a++) {
      int64_t s[4], e[4], tot = 0, mx = 0;
      for (int b = 0; b <= a; b++) {
        int64_t d = 0;
        for (int x = 0; x < HD; x++) d += q[a][hd * HD + x] * k[b][kv * HD + x];
        s[b] = d >> 6;
        if (b == 0 || s[b] > mx) mx = s[b];
      }
      for (int b = 0; b <= a; b++) { int64_t t = mx - s[b]; e[b] = EXPL[t > 64 ? 64 : t]; tot += e[b]; }
      int64_t wb[4];
      for (int b = 0; b <= a; b++) wb[b] = (e[b] * 256 + tot / 2) / tot;
      for (int x = 0; x < HD; x++) {
        int64_t acc = 0;
        for (int b = 0; b <= a; b++) acc += wb[b] * v[b][kv * HD + x];
        at[a][hd * HD + x] = sat8l(acc >> 8);
      }
    }
  }
  for (int m = 0; m < M; m++) {
    int64_t h2[DM];
    for (int j = 0; j < DM; j++) {
      int64_t a = 0;
      for (int i = 0; i < DM; i++) a += at[m][i] * w->wo[i * DM + j];
      h2[j] = sat8l(h[m][j] + (a >> 7));
    }
    for (int j = 0; j < DS; j++) {
      int64_t a = 0;
      for (int i = 0; i < DM; i++) a += h2[i] * w->down[i * DS + j];
      r[m][j] = sat8l(((int64_t)S[m][j] + sat8l(a >> 7)) >> 1);
    }
    int64_t a = 0;
    for (int j = 0; j < DS; j++) a += r[m][j] * w->sw[j];
    int64_t g = (a >> 7) + w->sb + 128;
    o->g[m] = (int)(g < 0 ? 0 : (g > 255 ? 255 : g));
    o->tau[m] = tau_of(o->g[m]);
    for (int kk = 0; kk < 4; kk++)
      for (int d = 0; d < DS; d++) {
        int64_t acc = 0;
        for (int j = 0; j < DS; j++) acc += (int64_t)w->gate[(kk * DS + d) * DS + j] * r[m][j];
        o->u[kk][m][d] = sat8l(acc >> 7);
      }
  }
  int64_t den = 0;
  for (int m = 0; m < M; m++) den += o->g[m] + 32;
  for (int d = 0; d < DS; d++) {
    int64_t num = 0;
    for (int m = 0; m < M; m++) num += (int64_t)(o->g[m] + 32) * o->u[0][m][d];
    o->slots[0][d] = sat8l(fdiv(2 * num + den, 2 * den));
    o->slots[1][d] = o->u[1][0][d];
    o->slots[2][d] = o->u[2][0][d];
  }
  for (int m = 1; m < M; m++) {
    int64_t dp = (int64_t)p[m] - (int64_t)p[m - 1];
    if (dp < 0) dp = 0;
    int64_t f = decay_q15(dp, o->tau[m]);
    int64_t al = (32768 - f) + (((int64_t)o->g[m] * f) >> 9);
    al = al < 4096 ? 4096 : (al > 32768 ? 32768 : al);
    for (int d = 0; d < DS; d++)
      o->slots[2][d] = sat8l(((32768 - al) * o->slots[2][d] + al * o->u[2][m][d] + 16384) >> 15);
  }
  for (int d = 0; d < DS; d++)
    o->slots[3][d] = echo ? sat8l((o->u[3][M - 1][d] + echo[d]) >> 1) : o->u[3][M - 1][d];
  for (int kk = 0; kk < 4; kk++) {
    int best = -1; int64_t bs = 0;
    for (int f = kk * 64; f < kk * 64 + 64; f++) {
      if (w->fs[f * 64] == 0) continue;
      int64_t d = 0;
      for (int x = 0; x < DS; x++) d += (int64_t)w->fb[f * DS + x] * o->slots[kk][x];
      if (best < 0 || d > bs) { best = f; bs = d; }
    }
    o->frame[kk] = best;
  }
  {
    int best = -1; int64_t bs = 0;
    for (int f = 0; f < 512; f++) {
      if (w->ps[f * 100] == 0) continue;
      int64_t d = 0;
      for (int x = 0; x < DS; x++) d += (int64_t)w->pb[f * DS + x] * o->slots[0][x];
      if (best < 0 || d > bs) { best = f; bs = d; }
    }
    o->phrase = best;
  }
  o->arc = -1;
  {
    int64_t bs = 0;
    for (int m = 1; m < M; m++) {
      int64_t d = 0;
      for (int x = 0; x < DS; x++) d += o->slots[2][x] * o->u[2][m][x];
      if (o->arc < 0 || d >= bs) { o->arc = m; bs = d; }
    }
  }
}

/* ------------------------------------------------------------------ SGRF + text */
typedef struct {
  const uint8_t *b; uint32_t ne, nv, na, eo, vo, ao;
} Sgrf;

static uint64_t rd64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

static long find_ent(const Sgrf *g, uint64_t th, uint64_t bh) {
  long lo = 0, hi = g->ne;
  while (lo < hi) {
    long mid = (lo + hi) / 2;
    const uint8_t *e = g->b + g->eo + mid * 64;
    uint64_t t = rd64(e), b = rd64(e + 8);
    if (t < th || (t == th && b < bh)) lo = mid + 1;
    else if (t == th && b == bh) return mid;
    else hi = mid;
  }
  return -1;
}

/* NUL-terminated copy of a fixed field */
static void field(char *dst, const uint8_t *src, int n) {
  int i = 0;
  while (i < n && src[i]) { dst[i] = (char)src[i]; i++; }
  dst[i] = 0;
}

/* vals: E R O A L B ; returns 0 when a used placeholder is empty */
static int fill(const char *tpl, const char *vals[6], char *out) {
  static const char *KEYS = "EROALB";
  int o = 0;
  for (int i = 0; tpl[i];) {
    if (tpl[i] == '%' && tpl[i + 1]) {
      const char *k = strchr(KEYS, tpl[i + 1]);
      if (k) {
        const char *v = vals[k - KEYS];
        if (!v[0]) return 0;
        size_t L = strlen(v); memcpy(out + o, v, L); o += (int)L; i += 2; continue;
      }
      if (tpl[i + 1] == '%') { out[o++] = '%'; i += 2; continue; }
    }
    out[o++] = tpl[i++];
  }
  out[o] = 0;
  return 1;
}

static void part(const Cspw *w, int idx, const char *dflt, const char *vals[6], char *out) {
  if (w && idx >= 0) {
    char tpl[65]; field(tpl, w->fs + idx * 64, 64);
    if (fill(tpl, vals, out)) return;
  }
  if (!fill(dflt, vals, out)) out[0] = 0;
}

/* returns -1 when no refresher, else byte length of text */
static int refresher(const Cspw *w, const Sgrf *g, uint64_t th, uint64_t bh, uint32_t cur, const char *word, char *text) {
  long ai = 0, ahi = g->na;
  while (ai < ahi) { long mid = (ai + ahi) / 2; if (rd64(g->b + g->ao + mid * 24) < th) ai = mid + 1; else ahi = mid; }
  for (; ai < (long)g->na && rd64(g->b + g->ao + ai * 24) == th; ai++) {
    uint32_t rev = rd32(g->b + g->ao + ai * 24 + 16);
    if (cur == 0 || cur >= rev) { th = rd64(g->b + g->ao + ai * 24 + 8); break; }
  }
  long ei = find_ent(g, th, bh);
  if (ei < 0) ei = find_ent(g, th, 0);
  if (ei < 0) return -1;
  const uint8_t *e = g->b + g->eo + ei * 64;
  uint32_t first = rd32(e + 16), start = rd32(e + 20);
  uint16_t cnt = rd16(e + 24);
  if (cur > 0 && first > cur) return -1;
  int *story = malloc(sizeof(int) * (cnt + 1)), *cross = malloc(sizeof(int) * (cnt + 1));
  int ns = 0, nx = 0;
  for (uint32_t i = start; i < start + cnt; i++) {
    const uint8_t *ev = g->b + g->vo + (size_t)i * 256;
    if (!(cur == 0 || rd32(ev) <= cur)) continue;
    if (ev[6] == 3) cross[nx++] = (int)i; else story[ns++] = (int)i;
  }
  int echo = nx ? cross[nx - 1] : -1;
  if (ns == 0) { memcpy(story, cross, sizeof(int) * nx); ns = nx; echo = -1; }
  if (ns == 0) { free(story); free(cross); return -1; }
  int sc[3], M = 0;
  sc[M++] = story[0];
  if (ns >= 3) {
    int bi = 1;
    for (int i = 2; i <= ns - 2; i++)
      if (g->b[g->vo + (size_t)story[i] * 256 + 7] > g->b[g->vo + (size_t)story[bi] * 256 + 7]) bi = i;
    sc[M++] = story[bi];
  }
  if (ns >= 2) sc[M++] = story[ns - 1];
  free(story); free(cross);
  const uint8_t *EV[3], *EC = echo >= 0 ? g->b + g->vo + (size_t)echo * 256 : NULL;
  for (int m = 0; m < M; m++) EV[m] = g->b + g->vo + (size_t)sc[m] * 256;
  WeaveOut wo;
  int arc_scene, fidx[4] = {-1, -1, -1, -1};
  char phrase[101] = "";
  if (w) {
    int8_t S[4][DS]; uint32_t pg[4];
    for (int m = 0; m < M; m++) { memcpy(S[m], EV[m] + 128, DS); pg[m] = rd16(EV[m] + 4); }
    weave(w, M, (const int8_t (*)[DS])S, pg, EC ? (const int8_t *)(EC + 128) : NULL, &wo);
    arc_scene = wo.arc;
    memcpy(fidx, wo.frame, sizeof fidx);
    if (wo.phrase >= 0) field(phrase, w->ps + wo.phrase * 100, 100);
  } else {
    arc_scene = M >= 2 ? M - 1 : -1;
  }
  char name[33], gl0[65], gla[65] = "", glb[65] = "", tl0[57], tla[57] = "", tlr[57], tle[57] = "";
  field(name, e + 28, 32);
  field(gl0, EV[0] + 8, 64); field(tl0, EV[0] + 72, 56);
  field(tlr, EV[M - 1] + 72, 56);
  if (arc_scene >= 0) { field(gla, EV[arc_scene] + 8, 64); field(tla, EV[arc_scene] + 72, 56); }
  if (EC) { field(glb, EC + 8, 64); field(tle, EC + 72, 56); }
  const char *E = name[0] ? name : word, *B = tle[0] ? tle : glb;
  const char *vals[6] = {E, phrase, gl0, gla, tlr, B};
  static char role[4096], origin[4096], arc[4096], ech[4096];
  arc[0] = 0; ech[0] = 0;
  part(w, fidx[0], "%E", vals, role);
  vals[4] = tl0; part(w, fidx[1], "%O", vals, origin);
  if (arc_scene >= 0) { vals[4] = tla[0] ? tla : tl0; part(w, fidx[2], "recently %A", vals, arc); }
  if (EC) { vals[4] = tle; part(w, fidx[3], "%B", vals, ech); }
  int n = sprintf(text, "Story: %s", role);
  if (origin[0] || arc[0]) {
    n += sprintf(text + n, " \xe2\x80\x94 ");
    if (origin[0]) n += sprintf(text + n, "%s", origin);
    if (origin[0] && arc[0]) n += sprintf(text + n, "; ");
    if (arc[0]) n += sprintf(text + n, "%s", arc);
  }
  char last = text[n - 1];
  if (last != '.' && last != '!' && last != '?') text[n++] = '.';
  text[n] = 0;
  if (ech[0]) n += sprintf(text + n, " | Echo: %s", ech);
  if (n > 511) {
    n = 511;
    int i = n - 1;
    while (i >= 0 && ((uint8_t)text[i] & 0xC0) == 0x80) i--;
    if (i >= 0 && i + u8len((uint8_t)text[i]) > n) n = i;
  }
  text[n] = 0;
  return n;
}

static uint8_t *slurp(const char *path, long *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *p = malloc(*len ? *len : 1);
  if (fread(p, 1, *len, f) != (size_t)*len) { fclose(f); free(p); return NULL; }
  fclose(f);
  return p;
}

static int hexbytes(const char *c, uint8_t *buf) {
  int n = 0;
  for (; c[0] && c[1] && c[0] != '\n' && c[0] != ' '; c += 2) { unsigned v; sscanf(c, "%2x", &v); buf[n++] = (uint8_t)v; }
  return n;
}

static int main_enc(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: ref enc model.bin engram.bin|- [nobigram]\n"); return 2; }
  long ml = 0, el = 0;
  uint8_t *mb = slurp(argv[2], &ml);
  if (!mb || ml != 95008) { fprintf(stderr, "bad model size %ld\n", ml); return 2; }
  uint32_t magic; uint16_t ver, d_byte, d_concept;
  memcpy(&magic, mb, 4); memcpy(&ver, mb + 4, 2); memcpy(&d_byte, mb + 6, 2); memcpy(&d_concept, mb + 8, 2);
  if (magic != 0x504C484E || ver != 1 || d_byte != 64 || d_concept != 128 || mb[10] != 2 || mb[11] != 2 ||
      mb[12] != 32 || mb[13] != 16) { fprintf(stderr, "bad header\n"); return 2; }
  float sc[4];
  memcpy(sc, mb + 14, 16);
  Plhn m;
  m.se = shift_of(sc[0]); m.sr = shift_of(sc[1]); m.sp = shift_of(sc[2]); m.sc = shift_of(sc[3]);
  const int8_t *p = (const int8_t *)(mb + 32);
  const int8_t **dst[] = {&m.emb, &m.c1dw, &m.c1pw, &m.c1g, &m.c2dw, &m.c2pw, &m.c2g, &m.rq, &m.rk, &m.cp,
                          &m.cdw, &m.cpw, &m.cg, &m.co};
  const int sizes[] = {256 * 64, 64 * 3, 64 * 64, 64 * 64, 64 * 3, 64 * 64, 64 * 64, 64 * 32, 64 * 32, 64 * 128,
                       128 * 3, 128 * 128, 128 * 128, 128 * 128};
  for (int i = 0; i < 14; i++) { *dst[i] = p; p += sizes[i]; }
  uint8_t *eng = NULL; uint32_t eslots = 0; int esh = 0;
  if (strcmp(argv[3], "-") != 0) {
    eng = slurp(argv[3], &el);
    uint32_t em, ev, sb; float es;
    memcpy(&em, eng, 4); memcpy(&ev, eng + 4, 4); memcpy(&eslots, eng + 8, 4); memcpy(&sb, eng + 12, 4);
    memcpy(&es, eng + 16, 4);
    if (em != 0x454E4752 || ev != 1 || sb != 64 || el < 512 + (long)eslots * 64) { fprintf(stderr, "bad engram\n"); return 2; }
    esh = shift_of(es);
  }
  int bigram = !(argc > 4 && strcmp(argv[4], "nobigram") == 0);
  static char line[8192];
  static uint8_t buf[4096];
  while (fgets(line, sizeof line, stdin)) {
    int n = hexbytes(line, buf);
    int8_t out[DC];
    encode(&m, eng, eslots, esh, bigram, buf, n, out);
    for (int d = 0; d < DC; d++) printf(d ? " %d" : "%d", out[d]);
    printf("\n");
  }
  return 0;
}

static int load_cspw(const char *path, Cspw *w) {
  long n = 0;
  uint8_t *b = slurp(path, &n);
  if (!b || !parse_cspw(b, n, w)) { fprintf(stderr, "bad cspw.bin\n"); return 0; }
  return 1;
}

/* stdin: M has_echo p0 p1 p2 p3 s[512] echo[128] ; stdout: g[4] tau[4] slots[512] frame[4] phrase arc */
static int main_cspw(int argc, char **argv) {
  Cspw w;
  if (argc < 3 || !load_cspw(argv[2], &w)) return 2;
  int M, he;
  while (scanf("%d %d", &M, &he) == 2) {
    uint32_t pg[4]; int8_t S[4][DS], ec[DS];
    for (int i = 0; i < 4; i++) if (scanf("%u", &pg[i]) != 1) return 3;
    for (int i = 0; i < 4 * DS; i++) { int v; if (scanf("%d", &v) != 1) return 3; S[i / DS][i % DS] = (int8_t)v; }
    for (int i = 0; i < DS; i++) { int v; if (scanf("%d", &v) != 1) return 3; ec[i] = (int8_t)v; }
    WeaveOut o;
    weave(&w, M, (const int8_t (*)[DS])S, pg, he ? ec : NULL, &o);
    for (int i = 0; i < 4; i++) printf("%d ", i < M ? o.g[i] : 0);
    for (int i = 0; i < 4; i++) printf("%d ", i < M ? o.tau[i] : 0);
    for (int i = 0; i < 4 * DS; i++) printf("%lld ", (long long)o.slots[i / DS][i % DS]);
    printf("%d %d %d %d %d %d\n", o.frame[0], o.frame[1], o.frame[2], o.frame[3], o.phrase, o.arc);
  }
  return 0;
}

/* stdin: term_hash book_hash cur word_hex|- ; stdout: text hex or '-' */
static int main_story(int argc, char **argv) {
  if (argc < 4) return 2;
  Cspw w, *wp = NULL;
  if (strcmp(argv[2], "-") != 0) { if (!load_cspw(argv[2], &w)) return 2; wp = &w; }
  long n = 0;
  uint8_t *b = slurp(argv[3], &n);
  if (!b || n < 512 || memcmp(b, "SGRF", 4) != 0) { fprintf(stderr, "bad sgrf\n"); return 2; }
  Sgrf g = {b, rd32(b + 8), rd32(b + 12), rd32(b + 16), rd32(b + 20), rd32(b + 24), rd32(b + 28)};
  unsigned long long th, bh; unsigned cur; char wh[1024];
  static char text[20000]; static uint8_t word[600];
  while (scanf("%llu %llu %u %1023s", &th, &bh, &cur, wh) == 4) {
    int wn = strcmp(wh, "-") == 0 ? 0 : hexbytes(wh, word);
    word[wn] = 0;
    int L = refresher(wp, &g, th, bh, cur, (const char *)word, text);
    if (L < 0) { printf("-\n"); continue; }
    for (int i = 0; i < L; i++) printf("%02x", (uint8_t)text[i]);
    printf("\n");
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "enc") == 0) return main_enc(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "cspw") == 0) return main_cspw(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "story") == 0) return main_story(argc, argv);
  fprintf(stderr, "usage: ref enc|cspw|story ...\n");
  return 2;
}
"""


def compile_ref(workdir: Path) -> Path:
    src = workdir / "plhn_cspw_ref.c"
    exe = workdir / "plhn_cspw_ref"
    src.write_text(C_SOURCE)
    subprocess.run(["cc", "-O2", "-std=c11", "-Wall", "-Werror", "-ffp-contract=off", "-o", str(exe), str(src), "-lm"],
                   check=True)
    return exe


def run_c(exe: Path, model: Path, engram: Path | None, inputs: list[bytes], bigram: bool = True):
    feed = "".join(b.hex() + "\n" for b in inputs)
    cmd = [str(exe), "enc", str(model), str(engram) if engram else "-"] + ([] if bigram else ["nobigram"])
    res = subprocess.run(cmd, input=feed, capture_output=True, text=True, check=True)
    return [[int(v) for v in line.split()] for line in res.stdout.splitlines()]


def run_c_cspw(exe: Path, cspw: Path, cases: list[dict]):
    lines = []
    for c in cases:
        M = len(c["scenes"])
        sc = np.zeros((4, 128), np.int64)
        sc[:M] = c["scenes"]
        pg = list(c["pages"]) + [0] * (4 - M)
        ec = c["echo"] if c["echo"] is not None else np.zeros(128, np.int64)
        lines.append(" ".join(map(str, [M, int(c["echo"] is not None), *pg, *sc.ravel().tolist(), *list(ec)])))
    res = subprocess.run([str(exe), "cspw", str(cspw)], input="\n".join(lines) + "\n", capture_output=True, text=True,
                         check=True)
    out = []
    for line in res.stdout.splitlines():
        v = [int(x) for x in line.split()]
        out.append({"g": v[0:4], "tau": v[4:8], "slots": np.array(v[8:520], np.int64).reshape(4, 128),
                    "frame_idx": v[520:524], "phrase_idx": v[524], "arc_scene": v[525]})
    return out


def run_c_story(exe: Path, cspw: Path | None, sgrf: Path, queries: list[tuple]):
    feed = "".join(f"{t} {b} {c} {w.encode('utf-8').hex() or '-'}\n" for t, b, c, w in queries)
    res = subprocess.run([str(exe), "story", str(cspw) if cspw else "-", str(sgrf)], input=feed, capture_output=True,
                         text=True, check=True)
    return [None if l == "-" else bytes.fromhex(l).decode("utf-8") for l in res.stdout.splitlines()]


def py_story(w, graph: CM.StoryGraph, q) -> str | None:
    t, b, cur, word = q
    hit = graph.lookup(t, b, cur)
    if hit is None:
        return None
    return CM.weave_refresher(w, hit[0], hit[1], word)[0]


def fuzz_inputs(n: int, seed: int, corpus: list[str]):
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz     ,.;!?'\"ABCXYZ0123456789的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太水万市眼体别处总才场师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放做系计或司利受光王果亲界及今京务制解各任至清物台象记边共风战干接它许八特觉望直服毛林题建南度统色字请交爱让认算论百吃义科怎元社术结六功指思非流每青管夫连远资队跟带花快条院变联言权往展该领传近留红治决周保达办运武半候七必城父强步完革深区即求品士转量空甚众技轻程告江语英基派满式李息写呢识极令黄德收脸钱党倒未持取设始版双历越史商千片容研像找友孩站广改议形委早房音火际则首单据导影失拿网香似斯专石若兵弟谁校读志飞观争究包组造落视济喜离虽坏兴切百🙂éñ—“”"
    idioms = ["画蛇添足", "一石二鸟", "守株待兔", "亡羊补牢", "对牛弹琴", "Harry Potter", "Elizabeth Bennet", "孙悟空"]
    out = []
    for i in range(n):
        kind = i % 6
        if kind == 0 and corpus:
            out.append(rng.choice(corpus).encode("utf-8")[: rng.randint(1, 400)])
        elif kind == 1:
            out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200))).encode("utf-8"))
        elif kind == 2:
            out.append(bytes(rng.randrange(256) for _ in range(rng.randint(0, 300))))  # malformed UTF-8
        elif kind == 3:
            out.append(bytes(rng.choice([0x80, 0xBF, 0xE4, 0x20, 0x61]) for _ in range(rng.randint(1, 260))))
        elif kind == 4:
            out.append("".join(rng.choice(idioms) + rng.choice(alphabet) for _ in range(rng.randint(1, 12))).encode())
        else:
            out.append(("".join(rng.choice(alphabet[:40]) for _ in range(rng.randint(1, 30))) * rng.randint(1, 20)).encode())
    return out


# ----------------------------------------------------------------------------------------------
# CS-PW fuzz material
# ----------------------------------------------------------------------------------------------

_TXT = ["the", "old", "river", "gate", "王", "府", "夜", "Pemberley", "Mr.", "sister", "雪", "—", "%", "!", "?", ".", "é"]


def _rtext(rng: random.Random, max_bytes: int, allow_empty: bool = True) -> str:
    if allow_empty and rng.random() < 0.15:
        return ""
    s = " ".join(rng.choice(_TXT) for _ in range(rng.randint(1, 14)))
    return CM.utf8_fit(s, max_bytes).decode("utf-8")


def _rtemplate(rng: random.Random, k: int) -> str:
    if rng.random() < 0.1:
        return ""
    parts = []
    for _ in range(rng.randint(1, 6)):
        r = rng.random()
        if r < 0.45:
            parts.append("%" + rng.choice("EROALB"))
        elif r < 0.5:
            parts.append(rng.choice(["%%", "%x", "%", "%é"]))
        else:
            parts.append(rng.choice(_TXT))
    s = " ".join(parts) + rng.choice(["", ".", "!", "?"])
    return CM.utf8_fit(s, CM.FRAME_BYTES - 1).decode("utf-8")


def random_cspw_blob(rng: random.Random) -> bytes:
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    t = {}
    for name, shape, _ in CM.CSPW_LAYOUT:
        mag = int(nrng.choice([8, 32, 127]))
        t[name] = nrng.integers(-mag, mag + 1, shape).astype(np.int8)
    return CM.pack_cspw(t, int(nrng.integers(-400, 401)), [_rtemplate(rng, i // 64) for i in range(256)],
                        [_rtext(rng, 99) for _ in range(512)])


def cspw_cases(n: int, seed: int, vec_pool: list | None = None):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        M = int(rng.integers(1, 5))
        if vec_pool and i % 2 == 0:
            idx = rng.integers(0, len(vec_pool), M + 1)
            sc = np.stack([np.asarray(vec_pool[j], np.int64) for j in idx[:M]])
            ec = np.asarray(vec_pool[idx[M]], np.int64)
        else:
            mag = int(rng.choice([20, 127]))
            sc = rng.integers(-mag, mag + 1, (M, 128))
            ec = rng.integers(-mag, mag + 1, 128)
        mode = i % 3
        if mode == 0:
            pages = np.sort(rng.integers(0, 4000, M))
        elif mode == 1:
            pages = rng.integers(0, 65536, M)  # non-monotone -> dp clamp
        else:
            pages = np.cumsum(rng.integers(0, 3, M))
        cases.append({"scenes": sc, "pages": [int(p) for p in pages], "echo": ec if rng.random() < 0.5 else None})
    return cases


def random_story(rng: random.Random):
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    terms = [rng.getrandbits(64) for _ in range(12)]
    books = [0, rng.getrandbits(64), rng.getrandbits(64)]
    ents, seen = [], set()
    for _ in range(18):
        key = (rng.choice(terms), rng.choice(books))
        if key in seen:
            continue
        seen.add(key)
        evs = []
        for _ in range(rng.randint(1, 9)):
            evs.append({"cutoff": rng.choice([rng.randint(1, 1 << 20), rng.randint(1 << 16, 6 << 16)]),
                        "chapter_or_page": rng.randint(0, 3000), "slot_hint": rng.choice([0, 1, 2, 2, 3]),
                        "salience_hint": rng.randint(0, 255), "event_gloss": _rtext(rng, 63),
                        "target_or_loc": _rtext(rng, 55), "vec": nrng.integers(-127, 128, 128).tolist()})
        first = min(e["cutoff"] for e in evs) if rng.random() < 0.7 else rng.randint(0, 6 << 16)
        ents.append({"term_hash": key[0], "book_path_hash": key[1], "first_cutoff": first,
                     "entity_kind": rng.randint(1, 4), "en_name_or_role": _rtext(rng, 31), "book_short": "bk",
                     "events": evs})
    aliases = [{"alias_hash": rng.choice(terms + [rng.getrandbits(64)]), "canonical_hash": rng.choice(terms),
                "reveal_cutoff": rng.randint(0, 6 << 16)} for _ in range(10)]
    blob = CM.pack_sgrf(ents, aliases)
    queries = []
    for _ in range(40):
        t = rng.choice(terms + [a["alias_hash"] for a in aliases] + [rng.getrandbits(64)])
        cur = rng.choice([0, rng.randint(1, 6 << 16), rng.randint(1, 1 << 20)])
        queries.append((t, rng.choice(books + [rng.getrandbits(64)]), cur, rng.choice(["", "word", "王府"])))
    return blob, queries


def graph_queries(graph: CM.StoryGraph, n: int, seed: int):
    rng = random.Random(seed)
    if not graph.entities:
        return []
    out = []
    for _ in range(n):
        e = rng.choice(graph.entities)
        evs = graph.events[e["event_start"]:e["event_start"] + e["event_count"]]
        cuts = [0, e["first_cutoff"], max(0, e["first_cutoff"] - 1)] + [v["cutoff"] for v in evs]
        cur = rng.choice(cuts) + rng.choice([0, 0, 1, -1 if cuts else 0])
        term = e["term_hash"]
        if graph.aliases and rng.random() < 0.3:
            term = rng.choice(graph.aliases)[0]
        book = e["book_path_hash"] if rng.random() < 0.8 else rng.getrandbits(64)
        out.append((term, book, max(0, cur), ""))
    return out


def _weave_eq(a: dict, b: dict) -> bool:
    M = len(b["g"])
    return (list(a["g"][:M]) == list(b["g"]) and list(a["tau"][:M]) == list(b["tau"])
            and np.array_equal(np.asarray(a["slots"]), np.asarray(b["slots"]))
            and list(a["frame_idx"]) == list(b["frame_idx"]) and a["phrase_idx"] == b["phrase_idx"]
            and a["arc_scene"] == b["arc_scene"])


# ----------------------------------------------------------------------------------------------
# Layout checks
# ----------------------------------------------------------------------------------------------


def check_layouts(model_blob: bytes, engram_blob: bytes | None, idx_path: Path | None):
    report = {}
    hdr = M.PLHN_HEADER.unpack_from(model_blob, 0)
    report["plhn"] = {"bytes": len(model_blob), "magic": hex(hdr[0]), "version": hdr[1], "dims": hdr[2:8],
                      "scales": hdr[8:12], "reserved": hdr[12]}
    assert len(model_blob) == M.PLHN_FILE_BYTES and hdr[0] == M.PLHN_MAGIC and hdr[12] == 0
    M.parse_plhn(model_blob)
    if engram_blob is not None:
        eh = M.parse_engr_header(engram_blob)
        assert len(engram_blob) == M.ENGR_FILE_BYTES and not any(engram_blob[20:512])
        slots = np.frombuffer(engram_blob, dtype=np.uint8, offset=512).reshape(-1, 64)
        hashes = slots[:, :8].copy().view("<u8").ravel()
        gl = slots[:, 32:64]
        has_gloss = gl[:, 0] != 0
        bad_nul = int((gl[:, 31] != 0).sum())
        bad_utf8 = 0
        for row in gl[has_gloss]:
            try:
                bytes(row).split(b"\0", 1)[0].decode("utf-8")
            except UnicodeDecodeError:
                bad_utf8 += 1
        report["engr"] = {"bytes": len(engram_blob), **eh, "slots_with_hash": int((hashes != 0).sum()),
                          "slots_with_gloss": int(has_gloss.sum()),
                          "nonzero_delta_slots": int((slots[:, 8:32] != 0).any(1).sum()),
                          "gloss_missing_nul": bad_nul, "gloss_bad_utf8": bad_utf8}
        assert bad_nul == 0 and bad_utf8 == 0
    if idx_path and idx_path.exists():
        blob = idx_path.read_bytes()
        magic, ver, count, dim = M.ICPT_HEADER.unpack_from(blob, 0)
        assert magic == M.ICPT_MAGIC and ver == 1 and dim == 128 and len(blob) == 16 + 192 * count
        norms, labels = [], set()
        for i in range(count):
            rec = M.ICPT_RECORD.unpack_from(blob, 16 + 192 * i)
            assert rec[3][-1] == 0 and rec[3].split(b"\0", 1)[0].decode("utf-8")
            labels.add(rec[3].split(b"\0", 1)[0].decode())
            norms.append(float(np.linalg.norm(np.array(rec[4:], dtype=np.float64))))
        report["icpt"] = {"bytes": len(blob), "count": count, "books": len(labels),
                          "vec_norm_min": min(norms) if norms else None, "vec_norm_max": max(norms) if norms else None}
    return report


def check_cspw_layout(blob: bytes) -> dict:
    w = CM.CspwWeights(blob)  # size, header, reserved-zero checks
    for base, n, size in ((CM.FRAME_STRINGS_OFFSET, CM.N_FRAMES, CM.FRAME_BYTES),
                          (CM.PHRASE_STRINGS_OFFSET, CM.N_PHRASES, CM.PHRASE_BYTES)):
        for i in range(n):
            raw = blob[base + i * size:base + (i + 1) * size]
            assert raw[-1] == 0, f"string {i} at {base} not NUL-terminated"
            raw.split(b"\0", 1)[0].decode("utf-8")
    need = ["%E", "%O", "%A", "%B"]
    typed = sum(1 for i, f in enumerate(w.frames) if f and need[i // 64] in f)
    return {"bytes": len(blob), "salience_b": w.salience_b, "frames": int(w.frame_ok.sum()),
            "phrases": int(w.phrase_ok.sum()), "frames_with_slot_placeholder": typed}


def check_sgrf_layout(blob: bytes) -> dict:
    g = CM.StoryGraph(blob)
    assert len(blob) % CM.SECTOR == 0
    nxt = 0
    for e in g.entities:
        assert e["event_start"] == nxt, "events not contiguous per entity"
        nxt += e["event_count"]
        cuts = [v["cutoff"] for v in g.events[e["event_start"]:nxt]]
        assert cuts == sorted(cuts), "event cutoffs not ascending"
    assert nxt == len(g.events)
    return {"bytes": len(blob), "entities": len(g.entities), "events": len(g.events), "aliases": len(g.aliases),
            "cross_events": sum(1 for v in g.events if v["slot_hint"] == CM.SLOT_HINT_CROSS)}


def read_golden_vectors(path: Path):
    blob = path.read_bytes()
    magic, ver, n, _ = struct.unpack_from("<4sIII", blob, 0)
    assert magic == b"PLGV" and ver == 1
    off, out = 16, []
    for _ in range(n):
        (L,) = struct.unpack_from("<H", blob, off)
        data = blob[off + 2:off + 2 + L]
        vec = list(struct.unpack_from("<128b", blob, off + 2 + L))
        out.append((data, vec))
        off += 2 + L + 128
    assert off == len(blob)
    return out


# ----------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--engram", type=Path)
    ap.add_argument("--idx", type=Path)
    ap.add_argument("--cspw", type=Path)
    ap.add_argument("--story", type=Path)
    ap.add_argument("--golden-dir", type=Path, default=ROOT / "golden")
    ap.add_argument("--ckpt", type=Path, help="optional train_cspw checkpoint: also check the torch QAT paths")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "train_corpus.jsonl")
    ap.add_argument("--fuzz", type=int, default=500, help="random inputs per engine")
    ap.add_argument("--py-samples", type=int, default=24, help="inputs also run through the slow pure-Python ref")
    ap.add_argument("--report", type=Path, default=ROOT / "parity_report.json")
    args = ap.parse_args(argv)
    md = args.models_dir
    model_p = args.model or md / "platonic-hnet.bin"
    engram_p = args.engram or md / "engram.bin"
    idx_p = args.idx  # optional ICPT concept index (per-reader, never published)
    cspw_p = args.cspw or md / "cspw.bin"
    story_p = args.story or md / "booksum-story-graph.idx"

    model_blob = model_p.read_bytes()
    engram_blob = engram_p.read_bytes() if engram_p.exists() else None
    report = {"layouts": check_layouts(model_blob, engram_blob, idx_p)}

    corpus = []
    if args.corpus.exists():
        with open(args.corpus, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 2000:
                    break
                try:
                    corpus.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    pass
    gpath = args.golden_dir / "golden.jsonl"
    golden = [json.loads(l) for l in gpath.read_text().splitlines() if l.strip()] if gpath.exists() else []
    inputs = [bytes.fromhex(g["hex"]) for g in golden] + fuzz_inputs(args.fuzz, 1234, corpus)

    # ---------------- ConceptEngine (PLHN + Engram incl. bigram)
    ref_np = M.PlhnIntNp(model_blob, engram_blob)
    ref_py = M.PlhnIntRef(model_blob, engram_blob)
    td_obj = tempfile.TemporaryDirectory()
    td = Path(td_obj.name)
    exe = compile_ref(td)
    c_out = run_c(exe, model_p, engram_p if engram_blob is not None else None, inputs)
    c_noeng = run_c(exe, model_p, None, inputs[:50])
    c_nobig = run_c(exe, model_p, engram_p, inputs[:100], bigram=False) if engram_blob is not None else []
    np_out = [ref_np.forward(b) for b in inputs]
    np_noeng = M.PlhnIntNp(model_blob, None)
    np_nobig = M.PlhnIntNp(model_blob, engram_blob, bigram=False) if engram_blob is not None else None
    mism_c = [i for i, (a, b) in enumerate(zip(c_out, np_out)) if a != list(b)]
    mism_noeng = [i for i, b in enumerate(inputs[:50]) if c_noeng[i] != list(np_noeng.forward(b))]
    mism_nobig = [i for i, b in enumerate(inputs[:len(c_nobig)]) if c_nobig[i] != list(np_nobig.forward(b))]
    bigram_hits = 0
    if np_nobig is not None:
        bigram_hits = sum(1 for i, b in enumerate(inputs[:len(c_nobig)]) if c_nobig[i] != list(np_out[i]))
    py_idx = list(range(min(args.py_samples, len(inputs))))
    mism_py = [i for i in py_idx if ref_py.forward(inputs[i]) != list(np_out[i])]
    mism_golden = [i for i, g in enumerate(golden) if g["vec"] != list(np_out[i])]
    gv_mism, gv_n = [], 0
    gvp = args.golden_dir / "golden_vectors.bin"
    if gvp.exists():
        gv = read_golden_vectors(gvp)
        gv_n = len(gv)
        gv_c = run_c(exe, model_p, engram_p if engram_blob is not None else None, [d for d, _ in gv])
        gv_mism = [i for i, (d, v) in enumerate(gv) if v != list(ref_np.forward(d)) or v != gv_c[i]]
    report["concept_parity"] = {
        "inputs": len(inputs), "golden": len(golden), "c_vs_numpy_mismatch": mism_c[:10],
        "c_vs_numpy_no_engram_mismatch": mism_noeng[:10], "c_vs_numpy_no_bigram_mismatch": mism_nobig[:10],
        "inputs_changed_by_bigram": bigram_hits, "py_samples": len(py_idx), "py_vs_numpy_mismatch": mism_py[:10],
        "golden_file_mismatch": mism_golden[:10], "golden_vectors": gv_n, "golden_vectors_mismatch": gv_mism[:10]}
    ok = not (mism_c or mism_noeng or mism_nobig or mism_py or mism_golden or gv_mism)

    # ---------------- CS-PW
    cspw_rep = {}
    w_dep = None
    if cspw_p.exists():
        cblob = cspw_p.read_bytes()
        cspw_rep["layout"] = check_cspw_layout(cblob)
        w_dep = CM.CspwWeights(cblob)
        pool = [list(v) for v in np_out[:200] if any(v)]
        cases = cspw_cases(args.fuzz, 77, pool)
        c_res = run_c_cspw(exe, cspw_p, cases)
        py_res = [CM.int8_cspw_weave(w_dep, c["scenes"], c["pages"], c["echo"]) for c in cases]
        mism = [i for i, (a, b) in enumerate(zip(c_res, py_res)) if not _weave_eq(a, b)]
        # random weights (saturation / sign coverage)
        rng = random.Random(99)
        mism_rw, n_rw = [], 0
        for k in range(4):
            rb = random_cspw_blob(rng)
            rp = td / f"rand{k}.cspw"
            rp.write_bytes(rb)
            rw = CM.CspwWeights(rb)
            rcases = cspw_cases(max(1, args.fuzz // 4), 1000 + k)
            rc = run_c_cspw(exe, rp, rcases)
            n_rw += len(rcases)
            mism_rw += [(k, i) for i, c in enumerate(rcases)
                        if not _weave_eq(rc[i], CM.int8_cspw_weave(rw, c["scenes"], c["pages"], c["echo"]))]
        cspw_rep["weave"] = {"cases": len(cases), "c_vs_python_mismatch": mism[:10], "random_weight_cases": n_rw,
                             "random_weight_mismatch": mism_rw[:10],
                             "frame_idx_hist": np.bincount([f // 64 for r in py_res for f in r["frame_idx"] if f >= 0],
                                                           minlength=4).tolist()}
        ok = ok and not mism and not mism_rw
        gp = args.golden_dir / "cspw_golden.bin"
        if gp.exists():
            gcases = CM.unpack_golden(gp.read_bytes())
            gc = run_c_cspw(exe, cspw_p, gcases)
            gm = [i for i, c in enumerate(gcases)
                  if not (_weave_eq(c, CM.int8_cspw_weave(w_dep, c["scenes"], c["pages"], c["echo"]))
                          and _weave_eq(gc[i], c))]
            cspw_rep["golden"] = {"cases": len(gcases), "mismatch": gm[:10]}
            ok = ok and not gm
        else:
            cspw_rep["golden"] = "missing"
            ok = False
        if args.ckpt:
            cspw_rep["torch_qat_vs_int_mismatch"] = torch_cspw_check(args.ckpt, w_dep, cases)
            ok = ok and not cspw_rep["torch_qat_vs_int_mismatch"]
    else:
        cspw_rep["layout"] = "missing"
        ok = False

    # ---------------- Story graph + device text
    story_rep = {}
    if story_p.exists():
        sblob = story_p.read_bytes()
        story_rep["layout"] = check_sgrf_layout(sblob)
        graph = CM.StoryGraph(sblob)
        qs = graph_queries(graph, args.fuzz, 5)
        mism_s = []
        for cp, w in ((cspw_p if w_dep is not None else None, w_dep), (None, None)):
            c_txt = run_c_story(exe, cp, story_p, qs)
            mism_s += [(cp is not None, i) for i, q in enumerate(qs) if c_txt[i] != py_story(w, graph, q)]
        found = sum(1 for q in qs if graph.lookup(q[0], q[1], q[2]) is not None)
        rng = random.Random(4242)
        mism_r, n_r = [], 0
        for k in range(max(1, args.fuzz // 40)):
            gb, rq = random_story(rng)
            gpth = td / f"rand{k}.idx"
            gpth.write_bytes(gb)
            rg = CM.StoryGraph(gb)
            rb = random_cspw_blob(rng)
            cpth = td / f"rand_s{k}.cspw"
            cpth.write_bytes(rb)
            rw = CM.CspwWeights(rb)
            c_txt = run_c_story(exe, cpth, gpth, rq)
            n_r += len(rq)
            mism_r += [(k, i) for i, q in enumerate(rq) if c_txt[i] != py_story(rw, rg, q)]
        story_rep["device_text"] = {"queries": len(qs), "found": found, "c_vs_python_mismatch": mism_s[:10],
                                    "random_graph_queries": n_r, "random_graph_mismatch": mism_r[:10]}
        if qs:
            sample = [py_story(w_dep, graph, q) for q in qs[:3]]
            story_rep["sample"] = sample
        ok = ok and not mism_s and not mism_r
    else:
        story_rep["layout"] = "missing"
        ok = False
    report["cspw"] = cspw_rep
    report["story"] = story_rep

    if args.ckpt:
        report["concept_parity"]["torch_qat_vs_numpy_mismatch"] = torch_encoder_check(args.ckpt, inputs, np_out,
                                                                                      ref_np.engram_hash)
        ok = ok and not report["concept_parity"]["torch_qat_vs_numpy_mismatch"]
    td_obj.cleanup()
    report["ok"] = ok
    args.report.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    return 0 if ok else 1


def _load_ckpt(path: Path):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.load(path, map_location="cpu", weights_only=False)


def torch_encoder_check(ckpt: Path, inputs, np_out, engram_hash):
    """Torch QAT integer path (bigram on, as on the device) with the deployed engram.bin hash table."""
    ck = _load_ckpt(ckpt)
    net = M.PlatonicEngramHNet(use_engram=True, use_bigram=True)
    missing, unexpected = net.load_state_dict(ck["encoder"] if "encoder" in ck else ck["state_dict"], strict=False)
    assert not (set(missing) - {"engram_hash"}) and not unexpected, (missing, unexpected)
    if engram_hash is not None:
        net.set_engram_hashes(engram_hash)
    net.eval()
    tq = []
    for i in range(0, len(inputs), 64):
        tq += net.encode_int(inputs[i:i + 64])
    return [i for i, (a, b) in enumerate(zip(tq, np_out)) if list(a) != list(b)][:10]


def torch_cspw_check(ckpt: Path, w: CM.CspwWeights, cases):
    import torch

    ck = _load_ckpt(ckpt)
    if "cspw" not in ck:
        return []
    cs = CM.ChronoSlotPointerWeaver(**ck["cspw_cfg"]).double()
    cs.load_state_dict(ck["cspw"])
    cs.quant = True
    B = len(cases)
    S = torch.zeros(B, 4, 128, dtype=torch.float64)
    P = torch.zeros(B, 4, dtype=torch.long)
    E = torch.zeros(B, 128, dtype=torch.float64)
    Mv = torch.tensor([len(c["scenes"]) for c in cases])
    he = torch.tensor([c["echo"] is not None for c in cases])
    for i, c in enumerate(cases):
        m = len(c["scenes"])
        S[i, :m] = torch.tensor(np.asarray(c["scenes"], np.float64))
        P[i, :m] = torch.tensor(c["pages"])
        if c["echo"] is not None:
            E[i] = torch.tensor(np.asarray(c["echo"], np.float64))
    with torch.no_grad():
        out = cs(S, P, Mv, E, he)
    bad = []
    for i, c in enumerate(cases):
        r = CM.int8_cspw_core(w, c["scenes"], c["pages"], c["echo"])
        m = len(c["scenes"])
        if not (np.array_equal(out["slots"][i].numpy().astype(np.int64), r["slots"])
                and out["g"][i, :m].numpy().astype(int).tolist() == r["g"]
                and out["tau"][i, :m].numpy().astype(int).tolist() == r["tau"]):
            bad.append(i)
    return bad[:10]


if __name__ == "__main__":
    sys.exit(main())
