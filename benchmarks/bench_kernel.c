/* Benchmark / segmentation harness around the bit-exact PLHN + CS-PW C reference.
 *
 * The reference source is ``verify_c_parity.C_SOURCE`` written verbatim to ``plhn_cspw_ref.c`` by
 * run_segmentation_efficiency_bench.py; it is included here with its ``main`` renamed, so every
 * timed call is the exact parity-checked kernel (encode(), block(), matvec(), weave()).
 *
 *   bench_kernel seg   model.bin  < lines.bin      boundary byte offsets of every '\n'-separated line
 *                                                   (whole line, no 256-byte cut: the router is causal and
 *                                                   streams; this is the firmware's per-byte decision)
 *   bench_kernel time  model.bin engram.bin|- cspw.bin|- reps < lines.bin
 *                                                   per-query wall time (CLOCK_MONOTONIC, ns) of
 *                                                   segment-only, full encode() and CS-PW weave(M=4)
 */
#define main plhn_ref_main
#include "plhn_cspw_ref.c"
#undef main

#include <time.h>

static int load_plhn(const char *path, Plhn *m) {
  long ml = 0;
  uint8_t *mb = slurp(path, &ml);
  if (!mb || ml != 95008) { fprintf(stderr, "bad model size %ld\n", ml); return 0; }
  uint32_t magic; uint16_t ver;
  memcpy(&magic, mb, 4); memcpy(&ver, mb + 4, 2);
  if (magic != 0x504C484E || ver != 1) { fprintf(stderr, "bad header\n"); return 0; }
  float sc[4];
  memcpy(sc, mb + 14, 16);
  m->se = shift_of(sc[0]); m->sr = shift_of(sc[1]); m->sp = shift_of(sc[2]); m->sc = shift_of(sc[3]);
  const int8_t *p = (const int8_t *)(mb + 32);
  const int8_t **dst[] = {&m->emb, &m->c1dw, &m->c1pw, &m->c1g, &m->c2dw, &m->c2pw, &m->c2g, &m->rq, &m->rk, &m->cp,
                          &m->cdw, &m->cpw, &m->cg, &m->co};
  const int sizes[] = {256 * 64, 64 * 3, 64 * 64, 64 * 64, 64 * 3, 64 * 64, 64 * 64, 64 * 32, 64 * 32, 64 * 128,
                       128 * 3, 128 * 128, 128 * 128, 128 * 128};
  for (int i = 0; i < 14; i++) { *dst[i] = p; p += sizes[i]; }
  return 1;
}

static int load_engram(const char *path, uint8_t **eng, uint32_t *eslots, int *esh) {
  *eng = NULL; *eslots = 0; *esh = 0;
  if (strcmp(path, "-") == 0) return 1;
  long el = 0;
  *eng = slurp(path, &el);
  uint32_t em, ev, sb; float es;
  memcpy(&em, *eng, 4); memcpy(&ev, *eng + 4, 4); memcpy(eslots, *eng + 8, 4); memcpy(&sb, *eng + 12, 4);
  memcpy(&es, *eng + 16, 4);
  if (em != 0x454E4752 || ev != 1 || sb != 64 || el < 512 + (long)*eslots * 64) { fprintf(stderr, "bad engram\n"); return 0; }
  *esh = shift_of(es);
  return 1;
}

/* Prelude + H-Net router over the whole line (same arithmetic and boundary rule as encode()). */
static int segment(const Plhn *m, const uint8_t *b, int n, int *starts) {
  if (n == 0) return 0;
  int32_t *h = malloc(sizeof(int32_t) * n * DB), *q = malloc(sizeof(int32_t) * n * DR),
          *k = malloc(sizeof(int32_t) * n * DR);
  for (int t = 0; t < n; t++)
    for (int d = 0; d < DB; d++) h[t * DB + d] = sat8(shr(m->emb[b[t] * DB + d], m->se - 5));
  block(h, n, DB, m->c1g, m->c1dw, m->c1pw, NULL, m->sp);
  block(h, n, DB, m->c2g, m->c2dw, m->c2pw, NULL, m->sp);
  for (int t = 0; t < n; t++) {
    matvec(h + t * DB, m->rq, DB, DR, m->sr, q + t * DR);
    matvec(h + t * DB, m->rk, DB, DR, m->sr, k + t * DR);
  }
  int nc = 0, len = 0;
  for (int t = 0; t < n; t++) {
    int boundary;
    if (t == 0 || len >= MAXSPAN) boundary = 1;
    else if ((b[t] & 0xC0) == 0x80) boundary = 0;
    else {
      int32_t dot = 0;
      for (int j = 0; j < DR; j++) dot += q[t * DR + j] * k[(t - 1) * DR + j];
      boundary = (len + u8len(b[t]) > MAXSPAN) || dot <= 0;
    }
    if (boundary) { starts[nc++] = t; len = 0; }
    len++;
  }
  free(h); free(q); free(k);
  return nc;
}

static uint8_t *read_stdin(long *n) {
  size_t cap = 1 << 20, len = 0;
  uint8_t *buf = malloc(cap);
  size_t r;
  while ((r = fread(buf + len, 1, cap - len, stdin)) > 0) {
    len += r;
    if (len == cap) { cap *= 2; buf = realloc(buf, cap); }
  }
  *n = (long)len;
  return buf;
}

static int split_lines(uint8_t *buf, long n, uint8_t ***lines, int **lens) {
  int cnt = 0, cap = 1024;
  *lines = malloc(sizeof(uint8_t *) * cap); *lens = malloc(sizeof(int) * cap);
  long s = 0;
  for (long i = 0; i <= n; i++) {
    if (i == n || buf[i] == '\n') {
      if (i == n && s == n) break;
      if (cnt == cap) { cap *= 2; *lines = realloc(*lines, sizeof(uint8_t *) * cap); *lens = realloc(*lens, sizeof(int) * cap); }
      (*lines)[cnt] = buf + s; (*lens)[cnt] = (int)(i - s); cnt++;
      s = i + 1;
    }
  }
  return cnt;
}

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static int cmp_d(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static int main_seg(int argc, char **argv) {
  Plhn m;
  if (argc < 3 || !load_plhn(argv[2], &m)) return 2;
  long n; uint8_t *buf = read_stdin(&n);
  uint8_t **lines; int *lens;
  int cnt = split_lines(buf, n, &lines, &lens);
  for (int i = 0; i < cnt; i++) {
    int *starts = malloc(sizeof(int) * (lens[i] + 1));
    int nc = segment(&m, lines[i], lens[i], starts);
    for (int c = 0; c < nc; c++) printf(c ? " %d" : "%d", starts[c]);
    printf("\n");
    free(starts);
  }
  return 0;
}

/* stdout: one line per query: n_bytes n_chunks seg_ns enc_ns (median over reps); then
 * "WEAVE <median ns> <mean ns> <calls>" for M=4 weave() calls over consecutive encoded queries. */
static int main_time(int argc, char **argv) {
  if (argc < 6) return 2;
  Plhn m;
  if (!load_plhn(argv[2], &m)) return 2;
  uint8_t *eng; uint32_t eslots; int esh;
  if (!load_engram(argv[3], &eng, &eslots, &esh)) return 2;
  Cspw w; int have_cspw = 0;
  if (strcmp(argv[4], "-") != 0) { if (!load_cspw(argv[4], &w)) return 2; have_cspw = 1; }
  int reps = atoi(argv[5]);
  long n; uint8_t *buf = read_stdin(&n);
  uint8_t **lines; int *lens;
  int cnt = split_lines(buf, n, &lines, &lens);
  int8_t (*vecs)[DC] = malloc(sizeof(int8_t[DC]) * (cnt + 4));
  double *ts = malloc(sizeof(double) * reps), *te = malloc(sizeof(double) * reps);
  int starts[MAXB + 1];
  volatile int sink = 0;
  for (int i = 0; i < cnt; i++) {
    int L = lens[i] > MAXB ? MAXB : lens[i];
    int nc = 0;
    for (int r = 0; r < reps; r++) {
      double t0 = now_ns();
      nc = segment(&m, lines[i], L, starts);
      double t1 = now_ns();
      encode(&m, eng, eslots, esh, 1, lines[i], L, vecs[i]);
      double t2 = now_ns();
      ts[r] = t1 - t0; te[r] = t2 - t1;
      sink += nc + vecs[i][0];
    }
    qsort(ts, reps, sizeof(double), cmp_d);
    qsort(te, reps, sizeof(double), cmp_d);
    printf("%d %d %.0f %.0f\n", L, nc, ts[reps / 2], te[reps / 2]);
  }
  if (have_cspw && cnt >= 4) {
    int calls = cnt - 3;
    double *tw = malloc(sizeof(double) * calls * reps), tot = 0;
    for (int i = 0; i < calls; i++) {
      int8_t S[4][DS];
      uint32_t pages[4];
      for (int j = 0; j < 4; j++) { memcpy(S[j], vecs[i + j], DS); pages[j] = 10 + 37 * j + (i % 11); }
      for (int r = 0; r < reps; r++) {
        WeaveOut o;
        double t0 = now_ns();
        weave(&w, 4, (const int8_t (*)[DS])S, pages, (i & 1) ? vecs[i] : NULL, &o);
        double t1 = now_ns();
        sink += o.phrase;
        tw[i * reps + r] = t1 - t0; tot += t1 - t0;
      }
    }
    qsort(tw, (size_t)calls * reps, sizeof(double), cmp_d);
    printf("WEAVE %.0f %.0f %d\n", tw[(size_t)calls * reps / 2], tot / ((double)calls * reps), calls * reps);
  }
  fprintf(stderr, "sink %d\n", sink);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "seg") == 0) return main_seg(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "time") == 0) return main_time(argc, argv);
  fprintf(stderr, "usage: bench_kernel seg|time ...\n");
  return 2;
}
