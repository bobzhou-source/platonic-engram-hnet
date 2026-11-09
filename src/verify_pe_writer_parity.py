"""PE-Writer bit-exact parity: NumPy reference vs batched torch integer reference (PyTorch QAT export) vs a
standalone C++17 reference vs the firmware ``ConceptEngine::generateMarginRiff`` kernel (compiled with the
host stubs in ``firmware/host_stubs``), over the 32 PWGD golden cases + ``--fuzz`` random inputs on
the exported model + ``--fuzz-models`` random PEWR/WCLS blobs (ties, empty buckets, malformed UTF-8, ...).

    python src/verify_pe_writer_parity.py --fuzz 200
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_writer_model as PW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "golden" / "pe_writer"
FIRMWARE = ROOT / "firmware"

# --------------------------------------------------------------------------- standalone C++ reference
CPP_REF = r"""
// Standalone PE-Writer v1 reference (ARCHITECTURE.md, PE-Writer section). stdin protocol per line:
//   cpage_hex cecho_hex subject_hex role_hex echo_hex      ('-' = empty string)
// stdout per line: plan_hex(384 B) b0,b1,b2 t0,...,t35 text_hex('-' if empty)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static const int D = 128, NV = 2048, T = 36, NB = 64, NS = 64, SB = 256;
static std::vector<uint8_t> M, Wc;

struct Lin { int S = 0, out = 0, in = 0; std::vector<int32_t> b; const int8_t* W = nullptr; };
static const int8_t *stepE, *cent, *tokE, *posE;
static Lin fc1, fc2, gateL, dwL, outL, head, copyL;
static size_t vocabOff;

static bool readFile(const char* p, std::vector<uint8_t>& v) {
  FILE* f = fopen(p, "rb"); if (!f) return false;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  v.resize(n); bool ok = fread(v.data(), 1, n, f) == (size_t)n; fclose(f); return ok;
}
static uint16_t u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static int32_t sh(int64_t a, int S) { return (int32_t)(S > 0 ? (a + (1LL << (S - 1))) >> S : a); }
static int32_t sat8(int64_t a) { return (int32_t)std::min<int64_t>(127, std::max<int64_t>(-127, a)); }

static bool parseLin(size_t& o, Lin& L, int out, int in) {
  float sc; memcpy(&sc, &M[o], 4);
  if (!(sc > 0.0f) || !std::isfinite(sc)) return false;
  long S = std::lround(-std::log2((double)sc));
  if (S < 0 || S > 30 || std::ldexp(1.0, (int)-S) != (double)sc) return false;
  L.S = (int)S; L.out = out; L.in = in; L.b.resize(out);
  for (int i = 0; i < out; ++i) L.b[i] = (int16_t)u16(&M[o + 4 + 2 * i]);
  L.W = (const int8_t*)&M[o + 4 + 2 * out];
  o += 4 + 2 * out + (size_t)out * in;
  return true;
}
static int32_t lin1(const Lin& L, int o, const int32_t* x) {
  int64_t acc = 0; const int8_t* w = L.W + (size_t)o * L.in;
  for (int i = 0; i < L.in; ++i) acc += (int64_t)w[i] * x[i];
  return sh(acc, L.S) + L.b[o];
}
static void normalize(const int64_t* r, int32_t* out) {
  int64_t n2 = 0; for (int d = 0; d < D; ++d) n2 += r[d] * r[d];
  uint64_t n = (uint64_t)std::sqrt((double)n2);
  while (n > 0 && n * n > (uint64_t)n2) --n;
  while ((n + 1) * (n + 1) <= (uint64_t)n2) ++n;
  for (int d = 0; d < D; ++d) {
    if (n == 0) { out[d] = 0; continue; }
    int64_t a = 127 * r[d], nn = (int64_t)n;
    int64_t q = a >= 0 ? (a + nn / 2) / nn : -((-a + nn / 2) / nn);
    out[d] = sat8(q);
  }
}
static bool cont(uint8_t b) { return (b & 0xC0) == 0x80; }
static bool decodeAt(const std::string& s, size_t i, uint32_t& cp, size_t& n) {
  if (i >= s.size()) return false;
  uint8_t b = (uint8_t)s[i];
  if (b < 0x80) { cp = b; n = 1; return true; }
  if (b >= 0xC0 && b < 0xE0) { n = 2; cp = b & 0x1F; }
  else if (b >= 0xE0 && b < 0xF0) { n = 3; cp = b & 0x0F; }
  else if (b >= 0xF0 && b < 0xF8) { n = 4; cp = b & 0x07; }
  else return false;
  if (i + n > s.size()) return false;
  for (size_t k = 1; k < n; ++k) { uint8_t c = (uint8_t)s[i + k]; if (!cont(c)) return false; cp = (cp << 6) | (c & 0x3F); }
  return true;
}
static bool cjk(uint32_t c) { return (c >= 0x2E80 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFF00 && c <= 0xFFEF); }
static bool lastCjk(const std::string& o) {
  if (o.empty()) return false;
  size_t i = o.size() - 1; int k = 0;
  while (i > 0 && k < 3 && cont((uint8_t)o[i])) { --i; ++k; }
  uint32_t cp; size_t n;
  return decodeAt(o, i, cp, n) && i + n == o.size() && cjk(cp);
}

static std::string unhex(const std::string& h) {
  if (h == "-") return "";
  std::string s; for (size_t i = 0; i + 1 < h.size(); i += 2) s.push_back((char)std::stoi(h.substr(i, 2), nullptr, 16));
  return s;
}
static std::string hex(const std::string& s) {
  static const char* H = "0123456789abcdef"; std::string o;
  for (unsigned char c : s) { o.push_back(H[c >> 4]); o.push_back(H[c & 15]); }
  return o.empty() ? "-" : o;
}

int main(int argc, char** argv) {
  if (argc < 3 || !readFile(argv[1], M) || !readFile(argv[2], Wc)) { fprintf(stderr, "usage: ref pewr wcls\n"); return 2; }
  if (M.size() != 641726 || memcmp(M.data(), "PEWR", 4) || u16(&M[4]) != 1 || u16(&M[6]) != 128 || u16(&M[8]) != 128 ||
      u16(&M[10]) != 3 || u16(&M[12]) != 2048 || u16(&M[14]) != 36 || u16(&M[16]) != 64 || u16(&M[18]) != 64) {
    fprintf(stderr, "bad PEWR\n"); return 3;
  }
  if (Wc.size() != 1048608 || memcmp(Wc.data(), "WCLS", 4) || u16(&Wc[4]) != 1 || u16(&Wc[6]) != 64 ||
      u16(&Wc[8]) != 64 || u16(&Wc[10]) != 256) { fprintf(stderr, "bad WCLS\n"); return 3; }
  size_t o = 32;
  stepE = (const int8_t*)&M[o]; o += 3 * D;
  if (!parseLin(o, fc1, D, D) || !parseLin(o, fc2, D, D)) return 4;
  cent = (const int8_t*)&M[o]; o += NB * D;
  tokE = (const int8_t*)&M[o]; o += (size_t)NV * D;
  posE = (const int8_t*)&M[o]; o += T * D;
  if (!parseLin(o, gateL, D, D) || !parseLin(o, dwL, D, 3) || !parseLin(o, outL, D, D) || !parseLin(o, head, NV, D) ||
      !parseLin(o, copyL, 1, D)) return 4;
  vocabOff = o;
  if (vocabOff + NV * 16 != M.size()) return 5;

  std::string a, b, cs, cr, ce;
  while (std::cin >> a >> b >> cs >> cr >> ce) {
    std::string pg = unhex(a), ec = unhex(b), subj = unhex(cs), role = unhex(cr), echo = unhex(ce);
    int32_t cpage[D], cecho[D], plan[3][D];
    for (int d = 0; d < D; ++d) { cpage[d] = (int8_t)pg[d]; cecho[d] = (int8_t)ec[d]; }
    // Stage 1
    const int32_t* prev = cpage;
    for (int s = 0; s < 3; ++s) {
      int32_t h[D], act[D]; int64_t r[D];
      for (int d = 0; d < D; ++d) h[d] = sat8((prev[d] + cecho[d] + stepE[s * D + d]) >> 1);
      for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < D; ++j) act[j] = std::min(127, std::max(0, lin1(fc1, j, h)));
        for (int d = 0; d < D; ++d) r[d] = (int64_t)h[d] + lin1(fc2, d, act);
        normalize(r, h);
      }
      memcpy(plan[s], h, sizeof(h)); prev = plan[s];
    }
    // Stage 2
    int bk[3]; int32_t rv[3][D] = {}; int32_t ctok[3][16] = {}; int32_t sim[3] = {0, 0, 0}; bool has[3] = {0, 0, 0};
    std::string ctext[3];
    for (int s = 0; s < 3; ++s) {
      int64_t best = 0; int bi = 0;
      for (int bb = 0; bb < NB; ++bb) {
        int64_t dt = 0; for (int d = 0; d < D; ++d) dt += (int64_t)cent[bb * D + d] * plan[s][d];
        if (bb == 0 || dt > best) { best = dt; bi = bb; }
      }
      bk[s] = bi;
      int64_t sb = 0; int sj = -1;
      for (int j = 0; j < NS; ++j) {
        const uint8_t* slot = &Wc[32 + ((size_t)bi * NS + j) * SB];
        if (slot[161] == 0) continue;
        int64_t dt = 0; for (int d = 0; d < D; ++d) dt += (int64_t)(int8_t)slot[d] * plan[s][d];
        if (sj < 0 || dt > sb) { sb = dt; sj = j; }
      }
      if (sj >= 0) {
        const uint8_t* slot = &Wc[32 + ((size_t)bi * NS + sj) * SB];
        for (int d = 0; d < D; ++d) rv[s][d] = (int8_t)slot[d];
        for (int k = 0; k < 16; ++k) { int t = u16(slot + 128 + 2 * k); ctok[s][k] = t >= NV ? 0 : t; }
        sim[s] = (int32_t)std::min<int64_t>(127, std::max<int64_t>(0, sb >> 7));
        size_t L = std::min<size_t>(slot[161], 93);
        std::string t((const char*)slot + 162, L);
        ctext[s] = t.substr(0, t.find('\0'));
        has[s] = true;
      }
    }
    // Stage 3
    int32_t y1[D] = {}, y2[D] = {}, u[D], y[D], z[D], hh[D];
    int prevTok = 1, cursor[3] = {0, 0, 0}; bool used[8] = {}; int out[T] = {};
    std::vector<int64_t> logit(NV);
    for (int t = 0; t < T; ++t) {
      int s = std::min(2, t / 12);
      for (int d = 0; d < D; ++d) u[d] = sat8(((int32_t)tokE[prevTok * D + d] + ((plan[s][d] + rv[s][d]) >> 1) + posE[t * D + d]) >> 1);
      for (int d = 0; d < D; ++d) {
        int32_t g = std::min(128, std::max(0, lin1(gateL, d, u) + 64));
        y[d] = (u[d] * g + 64) >> 7;
      }
      for (int d = 0; d < D; ++d) {
        const int8_t* w = dwL.W + d * 3;
        z[d] = sat8(sh((int64_t)w[0] * y2[d] + (int64_t)w[1] * y1[d] + (int64_t)w[2] * y[d], dwL.S) + dwL.b[d]);
      }
      memcpy(y2, y1, sizeof(y1)); memcpy(y1, y, sizeof(y));
      for (int d = 0; d < D; ++d) hh[d] = sat8((int64_t)u[d] + lin1(outL, d, z));
      for (int v = 0; v < NV; ++v) logit[v] = lin1(head, v, hh);
      int32_t gate = std::min(128, std::max(0, lin1(copyL, 0, hh) + 64));
      int tokc = cursor[s] < 16 ? ctok[s][cursor[s]] : 0;
      if (tokc > 0 && tokc < NV) logit[tokc] += (gate * sim[s]) >> 4;
      bool pen[NV] = {};
      for (int k = std::max(0, t - 6); k < t; ++k) if (out[k] >= 8) pen[out[k]] = true;
      for (int v = 8; v < NV; ++v) if (pen[v]) logit[v] -= 10000;
      bool excl[8] = {};
      excl[1] = true;
      for (int v = 2; v < 8; ++v) excl[v] = used[v];
      if (subj.empty()) excl[2] = true;
      if (echo.empty()) excl[4] = true;
      for (int k = 0; k < 3; ++k) if (!has[k]) excl[5 + k] = true;
      int yt = -1; int64_t bestL = 0;
      for (int v = 0; v < NV; ++v) {
        if (v < 8 && excl[v]) continue;
        if (yt < 0 || logit[v] > bestL) { bestL = logit[v]; yt = v; }
      }
      if (yt == 0) break;
      out[t] = yt; prevTok = yt;
      if (yt >= 2 && yt < 8) used[yt] = true;
      if (yt >= 5 && yt <= 7) cursor[yt - 5] = 16;
      else if (tokc > 0 && tokc < NV && yt == tokc) cursor[s] += 1;
    }
    // Render
    std::string res;
    for (int t = 0; t < T && out[t]; ++t) {
      int v = out[t]; std::string p;
      if (v == 2) p = subj;
      else if (v == 3) p = role.empty() ? "this concept" : role;
      else if (v == 4) p = echo;
      else if (v >= 5 && v <= 7) p = ctext[v - 5];
      else if (v >= 8) { const char* q = (const char*)&M[vocabOff + (size_t)v * 16]; p.assign(q, strnlen(q, 16)); }
      if (p.empty()) continue;
      if (p.size() >= 2 && p[0] == '#' && p[1] == '#') { res += p.substr(2); continue; }
      bool glue = res.empty() || strchr(",.;:!?)'", p[0]) != nullptr || res.back() == '(';
      if (!glue) { uint32_t cp; size_t n; glue = lastCjk(res) && decodeAt(p, 0, cp, n) && cjk(cp); }
      if (!glue) res.push_back(' ');
      res += p;
    }
    if (res.size() > 239) { size_t cut = 239; while (cut > 0 && cont((uint8_t)res[cut])) --cut; res.resize(cut); }
    std::string ph; for (int s = 0; s < 3; ++s) for (int d = 0; d < D; ++d) ph.push_back((char)(int8_t)plan[s][d]);
    std::cout << hex(ph) << ' ' << bk[0] << ',' << bk[1] << ',' << bk[2] << ' ';
    for (int t = 0; t < T; ++t) std::cout << out[t] << (t + 1 < T ? "," : "");
    std::cout << ' ' << hex(res) << '\n';
  }
  return 0;
}
"""

# --------------------------------------------------------------------------- firmware harness
FW_HARNESS = r"""
#include <cstdint>
#include <iostream>
#include <string>
#include "ConceptEngine.h"
static std::string unhex(const std::string& h) {
  if (h == "-") return "";
  std::string s; for (size_t i = 0; i + 1 < h.size(); i += 2) s.push_back((char)std::stoi(h.substr(i, 2), nullptr, 16));
  return s;
}
static std::string hex(const std::string& s) {
  static const char* H = "0123456789abcdef"; std::string o;
  for (unsigned char c : s) { o.push_back(H[c >> 4]); o.push_back(H[c & 15]); }
  return o.empty() ? "-" : o;
}
int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string dir = argv[1];
  std::string a, b, cs, cr, ce;
  while (std::cin >> a >> b >> cs >> cr >> ce) {
    std::string pg = unhex(a), ec = unhex(b), subj = unhex(cs), role = unhex(cr), echo = unhex(ce);
    int8_t cpage[128], cecho[128], plan[3][128] = {};
    for (int d = 0; d < 128; ++d) { cpage[d] = (int8_t)pg[d]; cecho[d] = (int8_t)ec[d]; }
    uint8_t buckets[3] = {};
    uint16_t toks[36] = {};
    std::string riff;
    ConceptEngine::generateMarginRiff(dir, cpage, cecho, subj.c_str(), role.c_str(), echo.c_str(), riff, plan, buckets,
                                      toks);
    std::string ph((const char*)&plan[0][0], 384);
    std::cout << hex(ph) << ' ' << int(buckets[0]) << ',' << int(buckets[1]) << ',' << int(buckets[2]) << ' ';
    for (int t = 0; t < 36; ++t) std::cout << toks[t] << (t + 1 < 36 ? "," : "");
    std::cout << ' ' << hex(riff) << '\n';
  }
  return 0;
}
"""


def compile_cpp(src: str, exe: Path, extra: list[str] | None = None, std: str = "c++17") -> None:
    p = exe.with_suffix(".cpp")
    p.write_text(src)
    cmd = ["clang++", f"-std={std}", "-O2", "-Wall", "-o", str(exe), str(p)] + (extra or [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed: {' '.join(cmd)}\n{r.stderr[-4000:]}")


def compile_firmware(fw: Path, exe: Path) -> bool:
    cpp = fw / "ConceptEngine.cpp"
    if not cpp.exists() or "generateMarginRiff" not in cpp.read_text(errors="replace"):
        return False
    compile_cpp(FW_HARNESS, exe, [str(cpp), "-I", str(fw / "host_stubs"), "-I", str(fw), "-w"], std="c++20")
    return True


def _line(c: dict) -> str:
    h = lambda b: b.hex() or "-"  # noqa: E731
    return " ".join([np.asarray(c["c_page"], np.int8).tobytes().hex(), np.asarray(c["c_echo"], np.int8).tobytes().hex(),
                     h(c["subject"]), h(c["role"]), h(c["echo_title"])]) + "\n"


def run_cpp(cmd: list[str], cases: list[dict]) -> list[dict]:
    r = subprocess.run(cmd, input="".join(_line(c) for c in cases), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}: {r.stderr[-2000:]}")
    out = []
    for ln in r.stdout.splitlines():
        ph, bk, tk, tx = ln.split(" ")
        out.append({"c_plan": np.frombuffer(bytes.fromhex(ph), np.int8).reshape(3, PW.D).astype(np.int64),
                    "buckets": [int(x) for x in bk.split(",")], "tokens": [int(x) for x in tk.split(",")],
                    "text": b"" if tx == "-" else bytes.fromhex(tx)})
    if len(out) != len(cases):
        raise RuntimeError(f"{cmd[0]}: {len(out)} outputs for {len(cases)} cases")
    return out


def same(a: dict, b: dict) -> list[str]:
    diff = []
    if not np.array_equal(np.asarray(a["c_plan"], np.int64), np.asarray(b["c_plan"], np.int64)):
        diff.append("c_plan")
    if list(a["buckets"]) != list(b["buckets"]):
        diff.append("buckets")
    if list(a["tokens"]) != list(b["tokens"]):
        diff.append("tokens")
    if bytes(a["text"]) != bytes(b["text"]):
        diff.append("text")
    return diff


# --------------------------------------------------------------------------- fuzz generators
_WORDS = ["reveals", "contrasts", "beneath", "the", "a", "of", ",", ".", ";", ":", "!", "?", ")", "(", "'s", "##ing",
          "##", "—", "orthodoxy", "survival", "Winston", "Julia", "孙", "悟", "空", "，", "。", "（", "é", "naïve",
          "über", "x" * 15, "\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e97", "", ""]
_STRS = ["", "Winston", "the clerk", "Moby-Dick", "孙悟空", "林黛玉", "红楼梦", "A" * 31, "(aside", "##frag",
         ", comma first", "naïve über", "Raskolnikov"]


def rand_str(rng: random.Random) -> bytes:
    s = rng.choice(_STRS).encode("utf-8")
    if rng.random() < 0.15:
        s = bytes(rng.randrange(1, 256) for _ in range(rng.randint(1, 31)))  # arbitrary bytes, NUL-free
    return s[:31]


def rand_inputs(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        def v():
            k = rng.random()
            if k < 0.06:
                return np.zeros(PW.D, np.int64)
            if k < 0.15:
                return nrng.integers(-128, 128, PW.D)
            if k < 0.2:
                return nrng.choice([-127, 127], PW.D)
            return PW.normalize_i8(nrng.integers(-64, 65, PW.D))
        out.append({"c_page": v(), "c_echo": v(), "subject": rand_str(rng), "role": rand_str(rng),
                    "echo_title": rand_str(rng)})
    return out


def rand_model(seed: int) -> tuple[bytes, bytes]:
    """Random PEWR + WCLS blobs that stress ties, saturation, copy boost, masks, empty buckets, odd UTF-8."""
    rng = np.random.default_rng(seed)
    pr = random.Random(seed)
    t = {}
    small = seed % 3 == 0  # tiny logits -> frequent ties
    for name, kind, shape in PW.PEWR_LAYOUT:
        if kind == "raw":
            t[name] = rng.integers(-127, 128, shape)
        else:
            S = int(rng.integers(10, 16)) if (small and name == "lm_head") else int(rng.integers(3, 10))
            W = np.clip(np.rint(rng.normal(0, 40, shape)), -127, 127).astype(np.int64)
            b = rng.integers(-300, 301, shape[0])
            if name == "lm_head":
                if small:
                    b[:] = 0  # all-zero bias + tiny logits -> argmax ties everywhere
                b[1:PW.N_SPECIAL] = rng.integers(0, 1500, PW.N_SPECIAL - 1)  # specials compete
                b[0] = int(rng.integers(-1000, 4000))  # EOS: never / mid-sequence / at t=0 (empty riff)
            t[name] = (S, b, W)
    t["bucket_centroids"] = PW.normalize_i8_rows(rng.integers(-60, 61, (PW.N_BUCKETS, PW.D)))
    if seed % 4 == 1:  # duplicate centroids -> bucket ties
        t["bucket_centroids"][1::2] = t["bucket_centroids"][0::2]
    vocab = [b""] * PW.N_SPECIAL + [pr.choice(_WORDS).encode("utf-8")[:15] for _ in range(PW.N_VOCAB - PW.N_SPECIAL)]
    pewr = PW.pack_pewr(t, vocab)
    vec = PW.normalize_i8_rows(rng.integers(-60, 61, (PW.N_BUCKETS * PW.N_SLOTS, PW.D))).reshape(
        PW.N_BUCKETS, PW.N_SLOTS, PW.D)
    if seed % 2 == 0:  # duplicate slot vectors -> slot ties
        vec[:, 1::2] = vec[:, 0::2]
    tok = rng.integers(0, PW.N_VOCAB + 300, (PW.N_BUCKETS, PW.N_SLOTS, PW.CLAUSE_TOKS)).astype(np.uint16)
    tok[rng.random(tok.shape) < 0.1] = 0
    texts, role = [], np.zeros((PW.N_BUCKETS, PW.N_SLOTS), np.uint8)
    for b in range(PW.N_BUCKETS):
        row = []
        empty_bucket = b % 9 == 4
        for j in range(PW.N_SLOTS):
            if empty_bucket or pr.random() < 0.2:
                row.append(b"")
                continue
            k = pr.random()
            if k < 0.1:
                s = ("隐藏的正统是生存的工具" * 4).encode("utf-8")[:93]
            elif k < 0.2:
                s = b"##" + b"x" * 30
            elif k < 0.3:
                s = bytes(pr.randrange(1, 256) for _ in range(pr.randint(1, 93)))
            else:
                s = " ".join(pr.choice(_WORDS) for _ in range(pr.randint(1, 14))).encode("utf-8")[:93]
            row.append(s if s else b"x")
        texts.append(row)
    wcls = PW.pack_wcls({"vec": vec.astype(np.int8), "tok": tok, "role": role, "text": texts})
    return pewr, wcls


# --------------------------------------------------------------------------- main
def compare_set(label, pewr, wcls, cases, impls, report, expected=None, max_show=5):
    np_ref = PW.PeWriterIntNp(pewr, wcls)
    t_ref = PW.PeWriterIntTorch(pewr, wcls)
    res = {"numpy": [np_ref.generate(c["c_page"], c["c_echo"], c["subject"], c["role"], c["echo_title"])
                     for c in cases]}
    res["torch"] = t_ref.generate(np.stack([np.asarray(c["c_page"]) for c in cases]),
                                  np.stack([np.asarray(c["c_echo"]) for c in cases]),
                                  [c["subject"] for c in cases], [c["role"] for c in cases],
                                  [c["echo_title"] for c in cases])
    for name, fn in impls.items():
        res[name] = fn(cases)
    if expected is not None:
        res["golden_expected"] = expected
    mism = {k: 0 for k in res if k != "numpy"}
    shown = 0
    for i in range(len(cases)):
        for k in mism:
            d = same(res["numpy"][i], res[k][i])
            if d:
                mism[k] += 1
                if shown < max_show:
                    shown += 1
                    print(f"  MISMATCH [{label}] case {i} numpy vs {k}: {d}\n"
                          f"    numpy tokens={res['numpy'][i]['tokens']}\n    {k:5} tokens={res[k][i]['tokens']}\n"
                          f"    numpy text={res['numpy'][i]['text']!r}\n    {k:5} text={res[k][i]['text']!r}")
    stats = {"cases": len(cases), "mismatches": mism,
             "emitted_tokens_mean": round(float(np.mean([sum(1 for v in r["tokens"] if v) for r in res["numpy"]])), 2),
             "cases_using_clause_token": sum(any(5 <= v <= 7 for v in r["tokens"]) for r in res["numpy"]),
             "cases_using_entity_token": sum(any(2 <= v <= 4 for v in r["tokens"]) for r in res["numpy"]),
             "cases_early_eos": sum(0 < sum(1 for v in r["tokens"] if v) < PW.T_MAX for r in res["numpy"]),
             "cases_empty_riff": sum(not r["text"] for r in res["numpy"]),
             "cases_text_truncated_239": sum(len(r["text"]) >= PW.TEXT_CAP - 4 for r in res["numpy"])}
    report[label] = stats
    print(f"[{label}] {json.dumps(stats)}")
    return sum(mism.values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "weights", help="pe-writer.bin + writer-clauses.bin")
    ap.add_argument("--dir", type=Path, default=GOLDEN_DIR, help="directory holding pe_writer_golden.bin")
    ap.add_argument("--fuzz", type=int, default=200, help="random inputs on the exported model")
    ap.add_argument("--fuzz-models", type=int, default=8, help="random PEWR/WCLS blobs (each gets --fuzz/4 inputs)")
    ap.add_argument("--firmware-dir", type=Path, default=FIRMWARE)
    ap.add_argument("--require-firmware", action="store_true", help="fail if the firmware kernel can't be built")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    pewr = (args.weights_dir / "pe-writer.bin").read_bytes()
    wcls = (args.weights_dir / "writer-clauses.bin").read_bytes()
    golden = PW.unpack_golden((args.dir / "pe_writer_golden.bin").read_bytes())
    PW.parse_pewr(pewr)
    PW.Wcls(wcls)
    report = {"layout": {"pe-writer.bin": len(pewr), "writer-clauses.bin": len(wcls), "golden_cases": len(golden),
                         "offsets": PW.OFFSETS, "vocab_str_offset": PW.VOCAB_STR_OFFSET}}
    assert len(golden) == 32, "PWGD must hold 32 cases"

    td = tempfile.TemporaryDirectory(prefix="pe-writer-parity-")
    tmp = Path(td.name)
    cref = tmp / "pe_writer_ref"
    compile_cpp(CPP_REF, cref)
    fw = tmp / "pe_writer_fw"
    have_fw = False
    try:
        have_fw = compile_firmware(args.firmware_dir, fw)
    except RuntimeError as e:
        print(f"firmware kernel build failed:\n{e}")
        if args.require_firmware:
            raise SystemExit(1)
    if not have_fw:
        print(f"note: firmware generateMarginRiff not available in {args.firmware_dir}; C++ reference only")
        if args.require_firmware:
            raise SystemExit(1)
    report["firmware_kernel"] = str(args.firmware_dir / "ConceptEngine.cpp") if have_fw else None

    def impls_for(pw_path: Path, wc_path: Path, model_dir: Path):
        d = {"cpp_ref": lambda cases: run_cpp([str(cref), str(pw_path), str(wc_path)], cases)}
        if have_fw:
            d["firmware"] = lambda cases: run_cpp([str(fw), str(model_dir)], cases)
        return d

    total = 0
    main_impls = impls_for(args.weights_dir / "pe-writer.bin", args.weights_dir / "writer-clauses.bin", args.weights_dir)
    exp = [{"c_plan": g["c_plan"], "buckets": g["buckets"], "tokens": g["tokens"], "text": g["text"]} for g in golden]
    total += compare_set("golden32", pewr, wcls, golden, main_impls, report, expected=exp)
    if args.fuzz > 0:
        total += compare_set(f"fuzz{args.fuzz}", pewr, wcls, rand_inputs(args.fuzz, args.seed), main_impls, report)
    per = max(1, args.fuzz // 4)
    for m in range(args.fuzz_models):
        md = tmp / f"model{m}"
        md.mkdir()
        pw, wc = rand_model(args.seed * 100 + m)
        (md / "pe-writer.bin").write_bytes(pw)
        (md / "writer-clauses.bin").write_bytes(wc)
        total += compare_set(f"random_model{m}", pw, wc, rand_inputs(per, args.seed * 31 + m),
                             impls_for(md / "pe-writer.bin", md / "writer-clauses.bin", md), report)
    report["total_mismatches"] = total
    out = args.report or (ROOT / "pe_writer_parity_report.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"TOTAL MISMATCHES: {total}  (implementations: numpy, torch-int, cpp_ref"
          f"{', firmware' if have_fw else ''}; report {out})")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
