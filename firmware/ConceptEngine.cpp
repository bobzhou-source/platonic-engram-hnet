#include "ConceptEngine.h"

#include <HalMemory.h>
#include <HalStorage.h>
#include <Logging.h>
#include <Memory.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ConceptEngine {
namespace {

// ---- PLHN v1 layout -------------------------------------------------------
constexpr uint32_t PLHN_MAGIC = 0x504C484E;  // "PLHN"
constexpr size_t HEADER_BYTES = 32;
constexpr size_t D_BYTE = 64;
constexpr size_t D_ROUTER = 32;
constexpr size_t SPAN_MAX = 12;
constexpr size_t ENGRAM_DIMS = 24;
constexpr int ACT_SHIFT = 5;  // activations are int8 at 2^-5

constexpr size_t OFF_EMBED = HEADER_BYTES;
constexpr size_t OFF_C1_DW = OFF_EMBED + 256 * D_BYTE;
constexpr size_t OFF_C1_PW = OFF_C1_DW + D_BYTE * 3;
constexpr size_t OFF_C1_GATE = OFF_C1_PW + D_BYTE * D_BYTE;
constexpr size_t OFF_C2_DW = OFF_C1_GATE + D_BYTE * D_BYTE;
constexpr size_t OFF_C2_PW = OFF_C2_DW + D_BYTE * 3;
constexpr size_t OFF_C2_GATE = OFF_C2_PW + D_BYTE * D_BYTE;
constexpr size_t OFF_ROUTER_Q = OFF_C2_GATE + D_BYTE * D_BYTE;
constexpr size_t OFF_ROUTER_K = OFF_ROUTER_Q + D_BYTE * D_ROUTER;
constexpr size_t OFF_CHUNK_PROJ = OFF_ROUTER_K + D_BYTE * D_ROUTER;
constexpr size_t OFF_CORE_DW = OFF_CHUNK_PROJ + D_BYTE * DIM;
constexpr size_t OFF_CORE_PW = OFF_CORE_DW + DIM * 3;
constexpr size_t OFF_CORE_GATE = OFF_CORE_PW + DIM * DIM;
constexpr size_t OFF_CORE_OUT = OFF_CORE_GATE + DIM * DIM;
constexpr size_t MODEL_BYTES = OFF_CORE_OUT + DIM * DIM;
static_assert(MODEL_BYTES == 95008, "PLHN v1 file size");

// ---- ENGR / ICPT ------------------------------------------------------------
constexpr uint32_t ENGR_MAGIC = 0x454E4752;  // "ENGR"
constexpr size_t ENGR_HEADER_BYTES = 512;
constexpr size_t ENGR_SLOT_BYTES = 64;
constexpr size_t ENGR_GLOSS_OFFSET = 8 + ENGRAM_DIMS;
constexpr size_t ENGR_GLOSS_BYTES = 32;

// A bigram span (two adjacent chunks) longer than this is not looked up.
constexpr size_t BIGRAM_MAX_BYTES = 18;

constexpr uint32_t ICPT_MAGIC = 0x49435054;  // "ICPT"
constexpr size_t ICPT_HEADER_BYTES = 16;
constexpr size_t ICPT_RECORD_BYTES = 192;
constexpr size_t ICPT_LABEL_OFFSET = 16;
constexpr size_t ICPT_LABEL_BYTES = 48;
constexpr size_t ICPT_VEC_OFFSET = 64;
constexpr size_t ICPT_BATCH = 16;
// 192 KB on the card; every lookup scans it once to query and once to append.
constexpr uint32_t MAX_EPISODIC_RECORDS = 1024;

constexpr size_t MAX_SCORED_SENSES = 24;

uint32_t readU32(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}
uint16_t readU16(const uint8_t* p) {
  uint16_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}
uint64_t readU64(const uint8_t* p) {
  uint64_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}
float readF32(const uint8_t* p) {
  float v;
  memcpy(&v, p, sizeof(v));
  return v;
}

// Header scales are exact powers of two: 2^-S.
bool scaleShift(const float scale, int& shift) {
  if (!(scale > 0.0f) || !std::isfinite(scale)) return false;
  const long s = std::lround(-std::log2(static_cast<double>(scale)));
  if (s < -24 || s > 30) return false;
  shift = static_cast<int>(s);
  return true;
}

int32_t sh(const int32_t a, const int s) {
  if (s > 0) return (a + (1 << (s - 1))) >> s;
  if (s < 0) return a << -s;
  return a;
}
int8_t sat8(const int32_t a) { return static_cast<int8_t>(std::clamp<int32_t>(a, -127, 127)); }
int8_t rq(const int32_t acc, const int s) { return sat8(sh(acc, s)); }

uint64_t isqrt(const uint64_t v) {
  auto r = static_cast<uint64_t>(std::sqrt(static_cast<double>(v)));
  while (r > 0 && r * r > v) --r;
  while ((r + 1) * (r + 1) <= v) ++r;
  return r;
}

int64_t rdiv(const int64_t a, const int64_t n) { return a >= 0 ? (a + n / 2) / n : -((-a + n / 2) / n); }

// Step 8 of the forward pass, shared with the fallback: exact L2 to int8.
void normalize(const int32_t* sums, int8_t out[DIM]) {
  int64_t n2 = 0;
  for (size_t d = 0; d < DIM; ++d) n2 += static_cast<int64_t>(sums[d]) * sums[d];
  const auto n = static_cast<int64_t>(isqrt(static_cast<uint64_t>(n2)));
  for (size_t d = 0; d < DIM; ++d) {
    out[d] = n == 0 ? 0 : static_cast<int8_t>(std::clamp<int64_t>(rdiv(127 * static_cast<int64_t>(sums[d]), n), -127, 127));
  }
}

// ---- Storage paths and model state -----------------------------------------
Paths paths;

struct Model {
  HalMemory::PsramBuffer weights;
  int se = 0, sr = 0, sp = 0, sc = 0;
  // Set once a present file failed to load (bad header, short read, no
  // PSRAM); a missing file is retried, so a model copied over USB is found.
  bool failed = false;
};
Model model;

const int8_t* weightsAt(const size_t offset) { return reinterpret_cast<const int8_t*>(model.weights.get() + offset); }

bool readFully(HalFile& file, uint8_t* out, const size_t bytes) {
  size_t done = 0;
  while (done < bytes) {
    const int got = file.read(out + done, bytes - done);
    if (got <= 0) return false;
    done += static_cast<size_t>(got);
  }
  return true;
}

bool loadModel() {
  if (model.weights) return true;
  if (model.failed || !Storage.exists(paths.model)) return false;
  model.failed = true;
  HalFile file;
  if (!Storage.openFileForRead("CPT", paths.model, file)) return false;
  if (file.fileSize() != MODEL_BYTES) {
    LOG_ERR("CPT", "Model size %u, expected %u", static_cast<unsigned>(file.fileSize()),
            static_cast<unsigned>(MODEL_BYTES));
    return false;
  }
  auto buffer = HalMemory::allocatePsram(MODEL_BYTES);
  if (!buffer) {
    LOG_ERR("CPT", "OOM: model weights");
    return false;
  }
  if (!readFully(file, buffer.get(), MODEL_BYTES)) return false;
  file.close();
  const uint8_t* h = buffer.get();
  if (readU32(h) != PLHN_MAGIC || readU16(h + 4) != 1 || readU16(h + 6) != D_BYTE || readU16(h + 8) != DIM ||
      h[10] != 2 || h[11] != 2 || h[12] != D_ROUTER || h[13] != 16) {
    LOG_ERR("CPT", "Bad model header");
    return false;
  }
  Model loaded;
  if (!scaleShift(readF32(h + 14), loaded.se) || !scaleShift(readF32(h + 18), loaded.sr) ||
      !scaleShift(readF32(h + 22), loaded.sp) || !scaleShift(readF32(h + 26), loaded.sc)) {
    LOG_ERR("CPT", "Bad model scales");
    return false;
  }
  loaded.weights = std::move(buffer);
  model = std::move(loaded);
  LOG_INF("CPT", "Concept model loaded");
  return true;
}

// ---- Engram file -------------------------------------------------------------
struct Engram {
  HalFile file;
  uint32_t slots = 0;
  int sg = 0;

  bool open(const char* path) {
    if (!path || !*path || !Storage.exists(path) || !Storage.openFileForRead("CPT", path, file)) return false;
    uint8_t header[20];
    if (file.read(header, sizeof(header)) != static_cast<int>(sizeof(header)) || readU32(header) != ENGR_MAGIC ||
        readU32(header + 4) != 1 || readU32(header + 12) != ENGR_SLOT_BYTES || !scaleShift(readF32(header + 16), sg)) {
      file.close();
      return false;
    }
    slots = readU32(header + 8);
    if (slots == 0 || file.fileSize() < ENGR_HEADER_BYTES + static_cast<size_t>(slots) * ENGR_SLOT_BYTES) {
      file.close();
      slots = 0;
      return false;
    }
    return true;
  }

  bool readSlot(const uint64_t hash, uint8_t slot[ENGR_SLOT_BYTES]) {
    if (slots == 0) return false;
    const size_t offset = ENGR_HEADER_BYTES + static_cast<size_t>(hash % slots) * ENGR_SLOT_BYTES;
    return file.seekSet(offset) && file.read(slot, ENGR_SLOT_BYTES) == static_cast<int>(ENGR_SLOT_BYTES);
  }
};

// ---- Forward pass --------------------------------------------------------------
struct BlockWeights {
  const int8_t* dw;
  const int8_t* pw;
  const int8_t* gate;
  const int8_t* out;  // core blocks only
  int s;
};

// Gated ShortConv (LFM2 double gate, hard-sigmoid in Q7), one step of a causal
// sequence; `y1`/`y2` hold the input-gated values one and two steps back.
template <size_t D>
struct BlockState {
  int8_t y1[D] = {};
  int8_t y2[D] = {};
};

template <size_t D>
void matVec(const int8_t* x, const int8_t* w, int32_t acc[D], const size_t in = D) {
  std::fill(acc, acc + D, 0);
  for (size_t i = 0; i < in; ++i) {
    const int32_t xi = x[i];
    if (xi == 0) continue;
    const int8_t* row = w + i * D;
    for (size_t j = 0; j < D; ++j) acc[j] += xi * row[j];
  }
}

template <size_t D>
void blockStep(const BlockWeights& w, BlockState<D>& state, int8_t x[D]) {
  int32_t acc[D];
  int16_t gs[D];
  int8_t y[D];
  int8_t z[D];
  matVec<D>(x, w.gate, acc);
  for (size_t c = 0; c < D; ++c) {
    gs[c] = static_cast<int16_t>(std::clamp<int32_t>(rq(acc[c], w.s) + 64, 0, 128));
    y[c] = static_cast<int8_t>((x[c] * gs[c] + 64) >> 7);
    const int8_t* tap = w.dw + c * 3;
    z[c] = rq(tap[0] * state.y2[c] + tap[1] * state.y1[c] + tap[2] * y[c], w.s);
  }
  memcpy(state.y2, state.y1, D);
  memcpy(state.y1, y, D);
  matVec<D>(z, w.pw, acc);
  int8_t o2[D];
  for (size_t c = 0; c < D; ++c) o2[c] = static_cast<int8_t>((rq(acc[c], w.s) * gs[c] + 64) >> 7);
  if (!w.out) {
    for (size_t c = 0; c < D; ++c) x[c] = sat8(x[c] + o2[c]);
    return;
  }
  matVec<D>(o2, w.out, acc);
  for (size_t c = 0; c < D; ++c) x[c] = sat8(x[c] + rq(acc[c], w.s));
}

bool continuation(const uint8_t b) { return (b & 0xC0) == 0x80; }
size_t utf8Length(const uint8_t b) { return b < 0x80 ? 1 : b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1; }

struct Forward {
  BlockWeights conv1{weightsAt(OFF_C1_DW), weightsAt(OFF_C1_PW), weightsAt(OFF_C1_GATE), nullptr, model.sp};
  BlockWeights conv2{weightsAt(OFF_C2_DW), weightsAt(OFF_C2_PW), weightsAt(OFF_C2_GATE), nullptr, model.sp};
  BlockWeights core{weightsAt(OFF_CORE_DW), weightsAt(OFF_CORE_PW), weightsAt(OFF_CORE_GATE), weightsAt(OFF_CORE_OUT),
                    model.sc};
  BlockState<D_BYTE> conv1State;
  BlockState<D_BYTE> conv2State;
  BlockState<DIM> coreLoop1;
  BlockState<DIM> coreLoop2;
  Engram engram;
  bool engramOpen = false;
  int32_t sums[DIM] = {};

  // `span` directly follows the previous chunk's `prevLen` bytes (0 for the
  // first chunk).
  void chunk(const uint8_t* span, const size_t len, const size_t prevLen, const int32_t pooled[D_BYTE]) {
    const int32_t recip = static_cast<int32_t>((65536 + len / 2) / len);
    int8_t m[D_BYTE];
    for (size_t d = 0; d < D_BYTE; ++d) m[d] = sat8((pooled[d] * recip + 32768) >> 16);
    int32_t acc[DIM];
    matVec<DIM>(m, weightsAt(OFF_CHUNK_PROJ), acc, D_BYTE);
    int8_t x[DIM];
    for (size_t d = 0; d < DIM; ++d) x[d] = rq(acc[d], model.sp);
    uint8_t slot[ENGR_SLOT_BYTES];
    const auto* delta = reinterpret_cast<const int8_t*>(slot + 8);
    if (engramOpen && engram.readSlot(fnv1a64({reinterpret_cast<const char*>(span), len}), slot)) {
      for (size_t d = 0; d < ENGRAM_DIMS; ++d) x[d] = sat8(x[d] + sat8(sh(delta[d], engram.sg - ACT_SHIFT)));
    }
    // Bigram span (previous chunk + this one): only an exact span-hash hit
    // counts, at half weight.
    if (engramOpen && prevLen > 0 && prevLen + len <= BIGRAM_MAX_BYTES) {
      const uint64_t h2 = fnv1a64({reinterpret_cast<const char*>(span - prevLen), prevLen + len});
      if (h2 != 0 && engram.readSlot(h2, slot) && readU64(slot) == h2) {
        for (size_t d = 0; d < ENGRAM_DIMS; ++d) {
          x[d] = sat8(x[d] + sat8(sh(static_cast<int32_t>(delta[d]) >> 1, engram.sg - ACT_SHIFT)));
        }
      }
    }
    blockStep<DIM>(core, coreLoop1, x);
    blockStep<DIM>(core, coreLoop2, x);
    for (size_t d = 0; d < DIM; ++d) sums[d] += x[d];
  }

  void run(const uint8_t* bytes, const size_t n) {
    const int8_t* embed = weightsAt(OFF_EMBED);
    int8_t kPrev[D_ROUTER] = {};
    int32_t pooled[D_BYTE] = {};
    size_t start = 0;
    size_t len = 0;
    size_t prevLen = 0;
    for (size_t t = 0; t < n; ++t) {
      const uint8_t b = bytes[t];
      int8_t h[D_BYTE];
      for (size_t d = 0; d < D_BYTE; ++d) h[d] = sat8(sh(embed[b * D_BYTE + d], model.se - ACT_SHIFT));
      blockStep<D_BYTE>(conv1, conv1State, h);
      blockStep<D_BYTE>(conv2, conv2State, h);
      int32_t qAcc[D_ROUTER];
      int32_t kAcc[D_ROUTER];
      matVec<D_ROUTER>(h, weightsAt(OFF_ROUTER_Q), qAcc, D_BYTE);
      matVec<D_ROUTER>(h, weightsAt(OFF_ROUTER_K), kAcc, D_BYTE);
      int32_t qk = 0;
      int8_t k[D_ROUTER];
      for (size_t d = 0; d < D_ROUTER; ++d) {
        qk += rq(qAcc[d], model.sr) * kPrev[d];
        k[d] = rq(kAcc[d], model.sr);
      }
      const bool boundary =
          t == 0 || len >= SPAN_MAX || (!continuation(b) && (len + utf8Length(b) > SPAN_MAX || qk <= 0));
      if (boundary) {
        if (t > 0) {
          chunk(bytes + start, len, prevLen, pooled);
          prevLen = len;
        }
        start = t;
        len = 0;
        std::fill(pooled, pooled + D_BYTE, 0);
      }
      for (size_t d = 0; d < D_BYTE; ++d) pooled[d] += h[d];
      ++len;
      memcpy(kPrev, k, D_ROUTER);
    }
    if (len > 0) chunk(bytes + start, len, prevLen, pooled);
  }
};

// ---- ICPT files ----------------------------------------------------------------
bool readIcptHeader(HalFile& file, uint32_t& count) {
  uint8_t header[ICPT_HEADER_BYTES];
  if (!file.seekSet(0) || file.read(header, sizeof(header)) != static_cast<int>(sizeof(header))) return false;
  if (readU32(header) != ICPT_MAGIC || readU32(header + 4) != 1 || readU32(header + 12) != DIM) return false;
  const size_t size = file.fileSize();
  const size_t stored = size > ICPT_HEADER_BYTES ? (size - ICPT_HEADER_BYTES) / ICPT_RECORD_BYTES : 0;
  count = std::min<uint32_t>(readU32(header + 8), static_cast<uint32_t>(stored));
  return true;
}

// Calls fn(index, record) for every record; fn returns false to stop.
template <typename Fn>
bool forEachRecord(HalFile& file, Fn&& fn) {
  uint32_t count = 0;
  if (!readIcptHeader(file, count)) return false;
  auto batch = makeUniqueNoThrow<uint8_t[]>(ICPT_BATCH * ICPT_RECORD_BYTES);
  if (!batch) {
    LOG_ERR("CPT", "OOM: index batch");
    return false;
  }
  for (uint32_t first = 0; first < count; first += ICPT_BATCH) {
    const uint32_t n = std::min<uint32_t>(ICPT_BATCH, count - first);
    const int bytes = static_cast<int>(n * ICPT_RECORD_BYTES);
    if (file.read(batch.get(), static_cast<size_t>(bytes)) != bytes) return false;
    for (uint32_t i = 0; i < n; ++i) {
      if (!fn(first + i, batch.get() + i * ICPT_RECORD_BYTES)) return true;
    }
  }
  return true;
}

template <typename Fn>
bool scanIndex(const char* path, Fn&& fn) {
  if (!path || !*path || !Storage.exists(path)) return false;
  HalFile file;
  if (!Storage.openFileForRead("CPT", path, file)) return false;
  const bool ok = forEachRecord(file, fn);
  file.close();
  return ok;
}

std::string recordLabel(const uint8_t* record) {
  const char* label = reinterpret_cast<const char*>(record + ICPT_LABEL_OFFSET);
  return std::string(label, strnlen(label, ICPT_LABEL_BYTES));
}

int32_t recordDot(const uint8_t* record, const int8_t query[DIM]) {
  return dot(reinterpret_cast<const int8_t*>(record + ICPT_VEC_OFFSET), query);
}

// The stem of a path's file name, cut to fit label[48] with its NUL without
// splitting a UTF-8 character.
std::string labelFor(std::string_view path) {
  const size_t slash = path.find_last_of('/');
  if (slash != std::string_view::npos) path.remove_prefix(slash + 1);
  const size_t dot = path.find_last_of('.');
  if (dot != std::string_view::npos && dot > 0) path = path.substr(0, dot);
  size_t len = std::min(path.size(), ICPT_LABEL_BYTES - 1);
  while (len > 0 && len < path.size() && continuation(static_cast<uint8_t>(path[len]))) --len;
  return std::string(path.substr(0, len));
}

void insertTop(std::vector<Match>& top, Match&& match, const size_t maxResults) {
  if (maxResults == 0) return;
  auto it = std::find_if(top.begin(), top.end(), [&](const Match& m) { return match.dot > m.dot; });
  if (top.size() >= maxResults && it == top.end()) return;
  top.insert(it, std::move(match));
  if (top.size() > maxResults) top.pop_back();
}

// ---- Sense parsing ----------------------------------------------------------------
std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}

// "12. text" / "3) text": the text after the marker, or npos when not a sense.
size_t numberedSenseStart(std::string_view line) {
  size_t i = 0;
  while (i < line.size() && line[i] == ' ') ++i;
  const size_t digits = i;
  while (i < line.size() && i - digits < 3 && line[i] >= '0' && line[i] <= '9') ++i;
  if (i == digits || i >= line.size() || (line[i] != '.' && line[i] != ')')) return std::string_view::npos;
  ++i;
  if (i < line.size() && line[i] != ' ') return std::string_view::npos;
  return i;
}

bool isLabelLine(std::string_view line) {
  line = trim(line);
  return line.size() >= 2 && line.front() == '(' && line.back() == ')';
}

struct Sense {
  size_t line;
  std::string_view text;    // trimmed sense text
  size_t label;             // line index of its label, or npos
  size_t segment;           // index of the ';' segment within its line (cedict)
};

struct ParsedDefinition {
  std::vector<std::string_view> lines;
  std::vector<Sense> senses;
  bool numbered = false;
  size_t headerEnd = 0;  // lines before this stay above the marked sense
};

std::vector<std::string_view> splitSegments(std::string_view line) {
  std::vector<std::string_view> segments;
  size_t start = 0;
  while (start <= line.size()) {
    const size_t semi = line.find(';', start);
    const size_t end = semi == std::string_view::npos ? line.size() : semi;
    segments.push_back(trim(line.substr(start, end - start)));
    if (semi == std::string_view::npos) break;
    start = semi + 1;
  }
  return segments;
}

ParsedDefinition parseSenses(std::string_view text) {
  ParsedDefinition parsed;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t nl = text.find('\n', start);
    const size_t end = nl == std::string_view::npos ? text.size() : nl;
    parsed.lines.push_back(text.substr(start, end - start));
    if (nl == std::string_view::npos) break;
    start = nl + 1;
  }
  const auto& lines = parsed.lines;
  const size_t npos = std::string_view::npos;
  size_t label = npos;
  for (size_t i = 0; i < lines.size(); ++i) {
    const size_t at = numberedSenseStart(lines[i]);
    if (at != npos) {
      const auto body = trim(lines[i].substr(at));
      if (!body.empty()) parsed.senses.push_back({i, body, label, 0});
    } else if (isLabelLine(lines[i])) {
      label = i;
    }
  }
  if (!parsed.senses.empty()) {
    parsed.numbered = true;
    const Sense& first = parsed.senses.front();
    parsed.headerEnd = first.label != npos ? first.label : first.line;
    return parsed;
  }
  // cedict: blocks of "header\nA; B; C" separated by blank lines, or a single
  // "A; B; C" line. Only "; " lists are senses: other lines may be prose.
  size_t blockStart = 0;
  size_t blocks = 0;
  std::vector<Sense> found;
  while (blockStart < lines.size()) {
    size_t blockEnd = blockStart;
    while (blockEnd < lines.size() && !trim(lines[blockEnd]).empty()) ++blockEnd;
    if (blockEnd > blockStart) {
      ++blocks;
      const bool hasHeader = blockEnd - blockStart > 1;
      for (size_t i = blockStart + (hasHeader ? 1 : 0); i < blockEnd; ++i) {
        if (lines[i].find(';') == std::string_view::npos) continue;
        const auto segments = splitSegments(lines[i]);
        for (size_t s = 0; s < segments.size(); ++s) {
          if (!segments[s].empty()) found.push_back({i, segments[s], hasHeader ? blockStart : npos, s});
        }
      }
    }
    blockStart = blockEnd + 1;
  }
  if (found.empty()) return parsed;
  parsed.headerEnd = found.front().line;
  // One block's header (its pinyin) already sits above the marked sense.
  if (blocks == 1)
    for (auto& sense : found) sense.label = npos;
  parsed.senses = std::move(found);
  return parsed;
}

// Index into parsed.senses of the best match, or npos with fewer than two.
size_t bestSenseIndex(const ParsedDefinition& parsed, const int8_t context[DIM]) {
  if (parsed.senses.size() < 2) return std::string_view::npos;
  size_t best = 0;
  int32_t bestDot = INT32_MIN;
  const size_t scored = std::min(parsed.senses.size(), MAX_SCORED_SENSES);
  for (size_t i = 0; i < scored; ++i) {
    int8_t vec[DIM];
    encodeText(parsed.senses[i].text, vec);
    const int32_t score = dot(context, vec);
    if (score > bestDot) {
      bestDot = score;
      best = i;
    }
  }
  return best;
}

// ---- CS-PW v2: cspw.bin ------------------------------------------------------------
// Integer kernel, bit-exact with the trainer's Python reference. Row-major
// [in][out] matrices, `>>` is an arithmetic (floor) shift, sat8 clamps to
// [-127, 127]. For M chronological scenes s_m (int8[128]) at pages p_m:
//  1. h_m = sat8((s_m @ up_proj) >> 7)                                  (192)
//  2. q = sat8((h @ gqa_q) >> 7) (4 heads x 48), k / v = sat8((h @ gqa_k|v) >> 7)
//     (2 KV heads x 48; query head h reads KV head h >> 1).
//  3. Causal Chunk-GQA: scene a attends b <= a with s_ab = (q_a . k_b) >> 6,
//     e_b = EXP_LUT_Q8[min(max s - s_ab, 64)], w_b = (256 e_b + S/2) / S
//     (S = sum e), attn_a = sat8((sum_b w_b v_b) >> 8).
//  4. h_m = sat8(h_m + ((attn_m @ gqa_o) >> 7)); v128 = sat8((h_m @ down_proj) >> 7);
//     r_m = sat8((s_m + v128) >> 1).
//  5. Salience g_m = clamp(((salience_w . r_m) >> 7) + salience_b + 128, 0, 255);
//     half-life tau_m = cspwHalfLife(g_m).
//  6. u^k_m[d] = sat8((sum_j slot_gate_w[k][d][j] r_m[j]) >> 7).
//  7. Slot 0 (Role): (g + 32)-weighted mean of u^0, rounded half up.
//     Slot 1 (Origin): u^1_0, locked to the first scene.
//     Slot 2 (Recent Arc): CfC over page gaps: f = cspwDecayQ15(p_m - p_{m-1},
//       tau_m), alpha = clamp(32768 - f + ((g_m f) >> 9), 4096, 32768),
//       arc = sat8(((32768 - alpha) arc + alpha u^2_m + 16384) >> 15).
//     Slot 3 (Cross-Book Echo): sat8((u^3_{M-1} + echo) >> 1), or u^3_{M-1}.
//  8. Per slot group k the frame in [64k, 64k + 64) with a non-empty string and
//     the largest dot with slot k (first wins ties); the phrase with the largest
//     dot with Slot 0; the arc scene m >= 1 with the largest dot(slot 2, u^2_m)
//     (last wins ties).
constexpr size_t CSPW_HEADER_BYTES = 512;
constexpr size_t CSPW_D_MODEL = 192;
constexpr size_t CSPW_HEADS = 4;
constexpr size_t CSPW_HEAD_DIM = 48;
constexpr size_t CSPW_KV = 96;
constexpr size_t CSPW_FRAMES = 256;
constexpr size_t CSPW_FRAME_GROUP = 64;
constexpr size_t CSPW_FRAME_CHARS = 64;
constexpr size_t CSPW_PHRASES = 512;
constexpr size_t CSPW_PHRASE_CHARS = 100;
constexpr size_t CSPW_OFF_UP = CSPW_HEADER_BYTES;
constexpr size_t CSPW_OFF_Q = CSPW_OFF_UP + DIM * CSPW_D_MODEL;
constexpr size_t CSPW_OFF_K = CSPW_OFF_Q + CSPW_D_MODEL * CSPW_D_MODEL;
constexpr size_t CSPW_OFF_V = CSPW_OFF_K + CSPW_D_MODEL * CSPW_KV;
constexpr size_t CSPW_OFF_O = CSPW_OFF_V + CSPW_D_MODEL * CSPW_KV;
constexpr size_t CSPW_OFF_DOWN = CSPW_OFF_O + CSPW_D_MODEL * CSPW_D_MODEL;
constexpr size_t CSPW_OFF_SAL_W = CSPW_OFF_DOWN + CSPW_D_MODEL * DIM;
constexpr size_t CSPW_OFF_SAL_B = CSPW_OFF_SAL_W + DIM;
constexpr size_t CSPW_OFF_GATE = CSPW_OFF_SAL_B + 128;
constexpr size_t CSPW_OFF_FRAME_BANK = CSPW_OFF_GATE + CSPW_SLOTS * DIM * DIM;
constexpr size_t CSPW_OFF_FRAME_STR = CSPW_OFF_FRAME_BANK + CSPW_FRAMES * DIM;
constexpr size_t CSPW_OFF_PHRASE_BANK = CSPW_OFF_FRAME_STR + CSPW_FRAMES * CSPW_FRAME_CHARS;
constexpr size_t CSPW_OFF_PHRASE_STR = CSPW_OFF_PHRASE_BANK + CSPW_PHRASES * DIM;
constexpr size_t CSPW_OFF_TAIL = CSPW_OFF_PHRASE_STR + CSPW_PHRASES * CSPW_PHRASE_CHARS;
constexpr size_t CSPW_BYTES = CSPW_OFF_TAIL + 1792;
static_assert(CSPW_OFF_Q == 25088 && CSPW_OFF_K == 61952 && CSPW_OFF_V == 80384 && CSPW_OFF_O == 98816 &&
                  CSPW_OFF_DOWN == 135680 && CSPW_OFF_SAL_W == 160256 && CSPW_OFF_SAL_B == 160384 &&
                  CSPW_OFF_GATE == 160512 && CSPW_OFF_FRAME_BANK == 226048 && CSPW_OFF_FRAME_STR == 258816 &&
                  CSPW_OFF_PHRASE_BANK == 275200 && CSPW_OFF_PHRASE_STR == 340736 && CSPW_OFF_TAIL == 391936,
              "CSPW v2 layout");
static_assert(CSPW_BYTES == 393728, "CSPW v2 file size");

// round(256 * 2^(-t/8)); index 64 (and beyond) weighs 0.
constexpr uint16_t EXP_LUT_Q8[65] = {256, 235, 215, 197, 181, 166, 152, 140, 128, 117, 108, 99, 91, 83, 76, 70, 64,
                                     59,  54,  49,  45,  41,  38,  35,  32,  29,  27,  25,  23, 21, 19, 17, 16, 15,
                                     13,  12,  11,  10,  10,  9,   8,   7,   7,   6,   6,   5,  5,  4,  4,  4,  3,
                                     3,   3,   3,   2,   2,   2,   2,   2,   2,   1,   1,   1,  1,  0};
// round(32768 * 2^(i/16)), i = 0..15.
constexpr uint32_t POW2_FRAC_Q15[16] = {32768, 34219, 35734, 37316, 38968, 40693, 42495, 44376,
                                        46341, 48393, 50535, 52773, 55109, 57549, 60097, 62757};
// round(32768 * 2^(-i/16)), i = 0..16: the 17-entry decay interpolation LUT.
constexpr uint32_t POW2_NEG_Q15[17] = {32768, 31379, 30048, 28774, 27554, 26386, 25268, 24196, 23170,
                                       22188, 21247, 20347, 19484, 18658, 17867, 17109, 16384};

struct Cspw {
  HalMemory::PsramBuffer weights;
  bool failed = false;  // as Model::failed
};
Cspw cspw;

const int8_t* cspwAt(const size_t offset) { return reinterpret_cast<const int8_t*>(cspw.weights.get() + offset); }

std::string_view fixedText(const void* field, const size_t capacity) {
  const auto* text = static_cast<const char*>(field);
  return {text, strnlen(text, capacity)};
}
std::string_view frameText(const size_t index) {
  return fixedText(cspw.weights.get() + CSPW_OFF_FRAME_STR + index * CSPW_FRAME_CHARS, CSPW_FRAME_CHARS);
}
std::string_view phraseText(const size_t index) {
  return fixedText(cspw.weights.get() + CSPW_OFF_PHRASE_STR + index * CSPW_PHRASE_CHARS, CSPW_PHRASE_CHARS);
}

bool loadCspw() {
  if (cspw.weights) return true;
  if (cspw.failed || !paths.cspw || !*paths.cspw || !Storage.exists(paths.cspw)) return false;
  cspw.failed = true;
  HalFile file;
  if (!Storage.openFileForRead("CPT", paths.cspw, file)) return false;
  if (file.fileSize() != CSPW_BYTES) {
    LOG_ERR("CPT", "CS-PW size %u, expected %u", static_cast<unsigned>(file.fileSize()),
            static_cast<unsigned>(CSPW_BYTES));
    return false;
  }
  auto buffer = HalMemory::allocatePsram(CSPW_BYTES);
  if (!buffer) {
    LOG_ERR("CPT", "OOM: CS-PW weights");
    return false;
  }
  if (!readFully(file, buffer.get(), CSPW_BYTES)) return false;
  file.close();
  const uint8_t* h = buffer.get();
  if (memcmp(h, "CSPW", 4) != 0 || readU16(h + 4) != 2 || readU32(h + 8) != CSPW_D_MODEL || readU32(h + 12) != DIM ||
      readU32(h + 16) != CSPW_SLOTS || readU32(h + 20) != CSPW_FRAMES || readU32(h + 24) != CSPW_PHRASES) {
    LOG_ERR("CPT", "Bad CS-PW header");
    return false;
  }
  cspw.weights = std::move(buffer);
  cspw.failed = false;
  LOG_INF("CPT", "CS-PW weaver loaded");
  return true;
}

// y[j] = sat8((sum_i x[i] * w[i][j]) >> 7) for a row-major [in][OUT] matrix.
template <size_t OUT>
void projectQ7(const int8_t* x, const size_t in, const int8_t* w, int8_t y[OUT]) {
  int32_t acc[OUT];
  matVec<OUT>(x, w, acc, in);
  for (size_t j = 0; j < OUT; ++j) y[j] = sat8(acc[j] >> 7);
}

int64_t floorDiv(const int64_t a, const int64_t b) {
  const int64_t q = a / b;
  return (a % b != 0 && a < 0) ? q - 1 : q;
}

struct CspwState {
  int8_t h[CSPW_MAX_SCENES][CSPW_D_MODEL];
  int8_t q[CSPW_MAX_SCENES][CSPW_D_MODEL];
  int8_t k[CSPW_MAX_SCENES][CSPW_KV];
  int8_t v[CSPW_MAX_SCENES][CSPW_KV];
  int8_t r[CSPW_MAX_SCENES][DIM];
  int8_t u[CSPW_SLOTS][CSPW_MAX_SCENES][DIM];
};

// Step 3 for scene `a`: the causal attention context over scenes 0..a.
void chunkGqa(const CspwState& st, const size_t a, int8_t attn[CSPW_D_MODEL]) {
  for (size_t head = 0; head < CSPW_HEADS; ++head) {
    const size_t qo = head * CSPW_HEAD_DIM;
    const size_t kvo = (head >> 1) * CSPW_HEAD_DIM;
    int32_t score[CSPW_MAX_SCENES];
    int32_t best = INT32_MIN;
    for (size_t b = 0; b <= a; ++b) {
      int32_t s = 0;
      for (size_t d = 0; d < CSPW_HEAD_DIM; ++d) s += st.q[a][qo + d] * st.k[b][kvo + d];
      score[b] = s >> 6;
      best = std::max(best, score[b]);
    }
    int32_t weight[CSPW_MAX_SCENES];
    int32_t sum = 0;
    for (size_t b = 0; b <= a; ++b) {
      weight[b] = EXP_LUT_Q8[std::min<int32_t>(best - score[b], 64)];
      sum += weight[b];
    }
    for (size_t b = 0; b <= a; ++b) weight[b] = (weight[b] * 256 + sum / 2) / sum;
    for (size_t d = 0; d < CSPW_HEAD_DIM; ++d) {
      int32_t c = 0;
      for (size_t b = 0; b <= a; ++b) c += weight[b] * st.v[b][kvo + d];
      attn[qo + d] = sat8(c >> 8);
    }
  }
}

// Best-dot row in [first, first + count) of a 128-wide bank, skipping rows
// whose string is empty; -1 when all are.
template <typename TextFn>
int16_t bestRow(const int8_t* bank, const size_t first, const size_t count, const int8_t query[DIM], TextFn&& text) {
  int16_t best = -1;
  int32_t bestDot = INT32_MIN;
  for (size_t i = first; i < first + count; ++i) {
    if (text(i).empty()) continue;
    const int32_t score = dot(bank + i * DIM, query);
    if (best < 0 || score > bestDot) {
      best = static_cast<int16_t>(i);
      bestDot = score;
    }
  }
  return best;
}

// ---- CS-PW v2: story-graph.idx --------------------------------------------------
constexpr size_t SGRF_HEADER_BYTES = 32;  // used part of the 512-byte header sector
constexpr size_t SGRF_ENTITY_BYTES = 64;
constexpr size_t SGRF_EVENT_BYTES = 256;
constexpr size_t SGRF_ALIAS_BYTES = 24;
constexpr size_t SGRF_NAME_OFFSET = 28;
constexpr size_t SGRF_NAME_BYTES = 32;
constexpr size_t SGRF_GLOSS_OFFSET = 8;
constexpr size_t SGRF_GLOSS_BYTES = 64;
constexpr size_t SGRF_TARGET_OFFSET = 72;
constexpr size_t SGRF_TARGET_BYTES = 56;
constexpr size_t SGRF_VEC_OFFSET = 128;
constexpr uint8_t SLOT_HINT_CROSS_BOOK = 3;
constexpr size_t SGRF_EVENT_BATCH = 8;

struct StoryGraph {
  HalFile file;
  uint32_t entities = 0, events = 0, aliases = 0;
  uint32_t entitiesOffset = 0, eventsOffset = 0, aliasesOffset = 0;

  bool open(const char* path) {
    if (!path || !*path || !Storage.exists(path) || !Storage.openFileForRead("CPT", path, file)) return false;
    uint8_t h[SGRF_HEADER_BYTES];
    if (file.read(h, sizeof(h)) != static_cast<int>(sizeof(h)) || memcmp(h, "SGRF", 4) != 0 || readU16(h + 4) != 1) {
      return false;
    }
    entities = readU32(h + 8);
    events = readU32(h + 12);
    aliases = readU32(h + 16);
    entitiesOffset = readU32(h + 20);
    eventsOffset = readU32(h + 24);
    aliasesOffset = readU32(h + 28);
    const uint64_t size = file.fileSize();
    const auto fits = [size](const uint32_t offset, const uint32_t count, const size_t bytes) {
      return count == 0 || static_cast<uint64_t>(offset) + static_cast<uint64_t>(count) * bytes <= size;
    };
    if (!fits(entitiesOffset, entities, SGRF_ENTITY_BYTES) || !fits(eventsOffset, events, SGRF_EVENT_BYTES) ||
        !fits(aliasesOffset, aliases, SGRF_ALIAS_BYTES)) {
      LOG_ERR("CPT", "Story graph truncated");
      return false;
    }
    return true;
  }

  bool readAt(const size_t offset, uint8_t* out, const size_t bytes) {
    return file.seekSet(offset) && file.read(out, bytes) == static_cast<int>(bytes);
  }

  // The canonical hash for `alias` when an alias row for it is revealed at
  // `cutoff`.
  bool resolveAlias(const uint64_t alias, const uint32_t cutoff, uint64_t& canonical) {
    uint32_t lo = 0, hi = aliases;
    uint8_t row[SGRF_ALIAS_BYTES];
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (!readAt(aliasesOffset + static_cast<size_t>(mid) * SGRF_ALIAS_BYTES, row, 8)) return false;
      if (readU64(row) < alias) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    for (uint32_t i = lo; i < aliases; ++i) {
      if (!readAt(aliasesOffset + static_cast<size_t>(i) * SGRF_ALIAS_BYTES, row, sizeof(row)) ||
          readU64(row) != alias) {
        return false;
      }
      if (cutoff == 0 || cutoff >= readU32(row + 16)) {
        canonical = readU64(row + 8);
        return true;
      }
    }
    return false;
  }

  bool findEntity(const uint64_t term, const uint64_t book, uint8_t record[SGRF_ENTITY_BYTES]) {
    uint32_t lo = 0, hi = entities;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (!readAt(entitiesOffset + static_cast<size_t>(mid) * SGRF_ENTITY_BYTES, record, 16)) return false;
      const uint64_t t = readU64(record);
      if (t < term || (t == term && readU64(record + 8) < book)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo < entities && readAt(entitiesOffset + static_cast<size_t>(lo) * SGRF_ENTITY_BYTES, record,
                                   SGRF_ENTITY_BYTES) &&
           readU64(record) == term && readU64(record + 8) == book;
  }
};

// Chronological picks among the eligible events of one kind: the first, the
// most salient strictly between first and last (earliest on ties), the last.
struct ScenePick {
  int32_t first = -1, middle = -1, last = -1;
  uint8_t middleSalience = 0, lastSalience = 0;

  void add(const int32_t index, const uint8_t salience) {
    if (first < 0) {
      first = index;
      return;
    }
    if (last >= 0 && (middle < 0 || lastSalience > middleSalience)) {
      middle = last;
      middleSalience = lastSalience;
    }
    last = index;
    lastSalience = salience;
  }

  size_t scenes(int32_t out[3]) const {
    size_t n = 0;
    for (const int32_t index : {first, middle, last})
      if (index >= 0) out[n++] = index;
    return n;
  }
};

struct StoryScene {
  uint16_t page = 0;
  std::string_view gloss;
  std::string_view target;
  uint8_t record[SGRF_EVENT_BYTES] = {};

  bool load(StoryGraph& graph, const uint32_t index) {
    if (!graph.readAt(graph.eventsOffset + static_cast<size_t>(index) * SGRF_EVENT_BYTES, record, sizeof(record))) {
      return false;
    }
    page = readU16(record + 4);
    gloss = fixedText(record + SGRF_GLOSS_OFFSET, SGRF_GLOSS_BYTES);
    target = fixedText(record + SGRF_TARGET_OFFSET, SGRF_TARGET_BYTES);
    return true;
  }
  const int8_t* vec() const { return reinterpret_cast<const int8_t*>(record + SGRF_VEC_OFFSET); }
};

struct WeaveWork {
  StoryScene scenes[3];
  StoryScene echo;
  uint8_t batch[SGRF_EVENT_BATCH * SGRF_EVENT_BYTES];
  int8_t vecs[3][DIM];
  uint32_t pages[3];
  ChronoSlots slots;
};

struct FrameFields {
  std::string_view e, r, o, a, l, b;
};

// `frame` with its placeholders substituted; false when one resolves empty.
bool fillFrame(std::string_view frame, const FrameFields& f, std::string& out) {
  out.clear();
  for (size_t i = 0; i < frame.size(); ++i) {
    if (frame[i] != '%' || i + 1 >= frame.size()) {
      out += frame[i];
      continue;
    }
    const char key = frame[i + 1];
    const std::string_view* value = key == 'E'   ? &f.e
                                    : key == 'R' ? &f.r
                                    : key == 'O' ? &f.o
                                    : key == 'A' ? &f.a
                                    : key == 'L' ? &f.l
                                    : key == 'B' ? &f.b
                                                 : nullptr;
    if (value) {
      if (value->empty()) return false;
      out.append(*value);
      ++i;
    } else if (key == '%') {
      out += '%';
      ++i;
    } else {
      out += '%';
    }
  }
  return true;
}

// One refresher part: the weaver's frame, else the default, else empty.
std::string weavePart(const int16_t frame, const char* fallback, const FrameFields& fields) {
  std::string out;
  if (frame >= 0 && cspw.weights && fillFrame(frameText(static_cast<size_t>(frame)), fields, out) && !out.empty()) {
    return out;
  }
  if (fillFrame(fallback, fields, out)) return out;
  return {};
}

// ---- PE-Writer v1: pe-writer.bin + writer-clauses.bin ------------------------------
// Exact integer semantics (pe-writer-kernel-addendum.md, shared with the
// trainer's NumPy reference and golden writer):
//   Linear L (f32 scale = 2^-S with S in 0..30, i16 bias[out], i8 W[out][in],
//     row = output): lin(L, x)[o] = sh(sum_i W[o][i] x[i], S) + bias[o].
//   gate_q7(z) = clamp(z + 64, 0, 128); normalize = the PLHN step-8 L2 to int8.
//   Planner, s = 0..2: h = sat8((c_prev + c_echo + step_embed[s]) >> 1); twice
//     h = normalize(h + lin(fc2, clamp(lin(fc1, h), 0, 127))).
//   Router: b_s = argmax_b dot(c_plan[s], centroid[b]) (lowest index on ties);
//     clause s = the valid slot (text_len > 0) of bucket b_s with the highest
//     dot with c_plan[s]; sim_q7 = clamp(dot >> 7, 0, 127).
//   Decoder, t = 0..35, phase s = min(2, t / 12), prev = <BOS> at t = 0:
//     u = sat8((tok_embed[prev] + ((c_plan[s] + r_s) >> 1) + pos_embed[t]) >> 1)
//     y = (u * gate_q7(lin(conv_in_gate, u)) + 64) >> 7
//     z = sat8(lin(conv_dw over y_{t-2}, y_{t-1}, y_t)); h = sat8(u + lin(conv_out_proj, z))
//     logits = lin(lm_head, h); + (gate_q7(lin(copy_gate_proj, h)) * sim_q7) >> 4 on
//     the clause's next token; -10000 on each token >= 8 emitted in the last six
//     steps; <BOS>, spent or unavailable entity/clause pointers masked; greedy.
constexpr uint32_t PEWR_MAGIC = 0x52574550;  // "PEWR"
constexpr uint32_t WCLS_MAGIC = 0x534C4357;  // "WCLS"
constexpr size_t PEWR_HEADER_BYTES = 32;
constexpr size_t PEWR_VOCAB = 2048;
constexpr size_t PEWR_VOCAB_CHARS = 16;
constexpr size_t PEWR_BUCKETS = 64;
constexpr size_t PEWR_SLOTS = 64;
constexpr size_t PEWR_STEP_TOKENS = 12;  // decoder tokens per thought phase
constexpr size_t PEWR_RECENT = 6;        // anti-repetition window
constexpr int32_t PEWR_REPEAT_PENALTY = 10000;
constexpr uint16_t TOK_EOS = 0;
constexpr uint16_t TOK_BOS = 1;
constexpr uint16_t TOK_SUBJECT = 2;
constexpr uint16_t TOK_ROLE = 3;
constexpr uint16_t TOK_ECHO = 4;
constexpr uint16_t TOK_CLAUSE0 = 5;
constexpr uint16_t TOK_FIRST_WORD = 8;

constexpr size_t linearBytes(const size_t out, const size_t in) { return 4 + 2 * out + out * in; }
constexpr size_t PEWR_OFF_STEP = PEWR_HEADER_BYTES;
constexpr size_t PEWR_OFF_FC1 = PEWR_OFF_STEP + PE_PLAN_STEPS * DIM;
constexpr size_t PEWR_OFF_FC2 = PEWR_OFF_FC1 + linearBytes(DIM, DIM);
constexpr size_t PEWR_OFF_CENTROIDS = PEWR_OFF_FC2 + linearBytes(DIM, DIM);
constexpr size_t PEWR_OFF_TOK = PEWR_OFF_CENTROIDS + PEWR_BUCKETS * DIM;
constexpr size_t PEWR_OFF_POS = PEWR_OFF_TOK + PEWR_VOCAB * DIM;
constexpr size_t PEWR_OFF_GATE = PEWR_OFF_POS + PE_MAX_TOKENS * DIM;
constexpr size_t PEWR_OFF_DW = PEWR_OFF_GATE + linearBytes(DIM, DIM);
constexpr size_t PEWR_OFF_OUT = PEWR_OFF_DW + linearBytes(DIM, 3);
constexpr size_t PEWR_OFF_LM = PEWR_OFF_OUT + linearBytes(DIM, DIM);
constexpr size_t PEWR_OFF_COPY = PEWR_OFF_LM + linearBytes(PEWR_VOCAB, DIM);
constexpr size_t PEWR_OFF_VOCAB = PEWR_OFF_COPY + linearBytes(1, DIM);
constexpr size_t PEWR_BYTES = PEWR_OFF_VOCAB + PEWR_VOCAB * PEWR_VOCAB_CHARS;
static_assert(PEWR_OFF_FC1 == 416 && PEWR_OFF_FC2 == 17060 && PEWR_OFF_CENTROIDS == 33704 && PEWR_OFF_TOK == 41896 &&
                  PEWR_OFF_POS == 304040 && PEWR_OFF_GATE == 308648 && PEWR_OFF_DW == 325292 &&
                  PEWR_OFF_OUT == 325936 && PEWR_OFF_LM == 342580 && PEWR_OFF_COPY == 608824 &&
                  PEWR_OFF_VOCAB == 608958,
              "PEWR v1 layout");
static_assert(PEWR_BYTES == 641726, "PEWR v1 file size");

constexpr size_t WCLS_HEADER_BYTES = 32;
constexpr size_t WCLS_SLOT_BYTES = 256;
constexpr size_t WCLS_TOKS_OFFSET = 128;
constexpr size_t WCLS_CLAUSE_TOKENS = 16;
constexpr size_t WCLS_TEXT_LEN_OFFSET = 161;
constexpr size_t WCLS_TEXT_OFFSET = 162;
constexpr size_t WCLS_TEXT_MAX = 93;
constexpr size_t WCLS_BUCKET_BYTES = PEWR_SLOTS * WCLS_SLOT_BYTES;
constexpr size_t WCLS_BYTES = WCLS_HEADER_BYTES + PEWR_BUCKETS * WCLS_BUCKET_BYTES;
static_assert(WCLS_BYTES == 1048608, "WCLS v1 file size");
constexpr size_t WCLS_BATCH_SLOTS = 16;  // 4 KB of the 16 KB bucket per read

// One int8 linear layer inside the loaded pe-writer.bin. The f32 scale is
// held exactly as mant * 2^-shift (mant < 2^24), so the rounding is integer.
struct PeLinear {
  const int8_t* w = nullptr;
  const uint8_t* bias = nullptr;  // i16 LE, unaligned
  int s = 0;
  size_t out = 0, in = 0;

  bool init(const uint8_t* base, const size_t offset, const size_t outDim, const size_t inDim) {
    const float scale = readF32(base + offset);
    if (!scaleShift(scale, s) || s < 0 || std::ldexp(1.0f, -s) != scale) return false;
    bias = base + offset + 4;
    w = reinterpret_cast<const int8_t*>(base + offset + 4 + 2 * outDim);
    out = outDim;
    in = inDim;
    return true;
  }

  int32_t biasAt(const size_t o) const { return static_cast<int16_t>(readU16(bias + 2 * o)); }

  void apply(const int8_t* x, int32_t* y) const {
    for (size_t o = 0; o < out; ++o) {
      const int8_t* row = w + o * in;
      int32_t acc = 0;
      for (size_t i = 0; i < in; ++i) acc += static_cast<int32_t>(row[i]) * x[i];
      y[o] = sh(acc, s) + biasAt(o);
    }
  }
};

int32_t gateQ7(const int32_t z) { return std::clamp<int32_t>(z + 64, 0, 128); }

struct PeWriter {
  const uint8_t* base = nullptr;
  PeLinear fc1, fc2, gate, dw, outProj, lm, copy;

  const int8_t* at(const size_t offset) const { return reinterpret_cast<const int8_t*>(base + offset); }
  std::string_view vocab(const size_t token) const {
    return fixedText(base + PEWR_OFF_VOCAB + token * PEWR_VOCAB_CHARS, PEWR_VOCAB_CHARS);
  }

  bool init(const uint8_t* bytes) {
    base = bytes;
    return readU32(bytes) == PEWR_MAGIC && readU16(bytes + 4) == 1 && readU16(bytes + 6) == DIM &&
           readU16(bytes + 8) == DIM && readU16(bytes + 10) == PE_PLAN_STEPS && readU16(bytes + 12) == PEWR_VOCAB &&
           readU16(bytes + 14) == PE_MAX_TOKENS && readU16(bytes + 16) == PEWR_BUCKETS &&
           readU16(bytes + 18) == PEWR_SLOTS && fc1.init(bytes, PEWR_OFF_FC1, DIM, DIM) &&
           fc2.init(bytes, PEWR_OFF_FC2, DIM, DIM) && gate.init(bytes, PEWR_OFF_GATE, DIM, DIM) &&
           dw.init(bytes, PEWR_OFF_DW, DIM, 3) && outProj.init(bytes, PEWR_OFF_OUT, DIM, DIM) &&
           lm.init(bytes, PEWR_OFF_LM, PEWR_VOCAB, DIM) && copy.init(bytes, PEWR_OFF_COPY, 1, DIM);
  }
};

// The retrieved clause for one thought step (empty when its bucket has no
// valid slot).
struct PeClause {
  bool valid = false;
  int8_t vec[DIM] = {};
  uint16_t toks[WCLS_CLAUSE_TOKENS] = {};
  int32_t simQ7 = 0;
  char text[WCLS_TEXT_MAX + 1] = {};
};

struct PeWork {
  int8_t plan[PE_PLAN_STEPS][DIM];
  uint8_t buckets[PE_PLAN_STEPS];
  PeClause clauses[PE_PLAN_STEPS];
  uint16_t tokens[PE_MAX_TOKENS];
  int32_t logits[PEWR_VOCAB];
  uint8_t batch[WCLS_BATCH_SLOTS * WCLS_SLOT_BYTES];
};

void runPlanner(const PeWriter& pe, const int8_t cPage[DIM], const int8_t cEcho[DIM], PeWork& work) {
  const int8_t* prev = cPage;
  int32_t a32[DIM], f32[DIM];
  int8_t x[DIM], a[DIM];
  for (size_t s = 0; s < PE_PLAN_STEPS; ++s) {
    const int8_t* step = pe.at(PEWR_OFF_STEP + s * DIM);
    for (size_t d = 0; d < DIM; ++d) x[d] = sat8((prev[d] + cEcho[d] + step[d]) >> 1);
    for (int loop = 0; loop < 2; ++loop) {
      pe.fc1.apply(x, a32);
      for (size_t d = 0; d < DIM; ++d) a[d] = static_cast<int8_t>(std::clamp<int32_t>(a32[d], 0, 127));
      pe.fc2.apply(a, f32);
      for (size_t d = 0; d < DIM; ++d) f32[d] += x[d];
      normalize(f32, x);
    }
    memcpy(work.plan[s], x, DIM);
    prev = work.plan[s];
  }
}

uint8_t routeBucket(const PeWriter& pe, const int8_t plan[DIM]) {
  uint8_t best = 0;
  int32_t bestDot = INT32_MIN;
  for (size_t b = 0; b < PEWR_BUCKETS; ++b) {
    const int32_t score = dot(plan, pe.at(PEWR_OFF_CENTROIDS + b * DIM));
    if (score > bestDot) {
      bestDot = score;
      best = static_cast<uint8_t>(b);
    }
  }
  return best;
}

// One seek into writer-clauses.bin: the bucket's 64 slots stream through the
// 4 KB batch buffer and only the best one is kept.
bool retrieveClause(HalFile& file, const uint8_t bucket, const int8_t plan[DIM], PeWork& work, PeClause& out) {
  out = PeClause{};
  if (!file.seekSet(WCLS_HEADER_BYTES + static_cast<size_t>(bucket) * WCLS_BUCKET_BYTES)) return false;
  int32_t bestDot = INT32_MIN;
  for (size_t first = 0; first < PEWR_SLOTS; first += WCLS_BATCH_SLOTS) {
    if (!readFully(file, work.batch, sizeof(work.batch))) return false;
    for (size_t i = 0; i < WCLS_BATCH_SLOTS; ++i) {
      const uint8_t* slot = work.batch + i * WCLS_SLOT_BYTES;
      const uint8_t textLen = slot[WCLS_TEXT_LEN_OFFSET];
      if (textLen == 0) continue;
      const auto* vec = reinterpret_cast<const int8_t*>(slot);
      const int32_t score = dot(plan, vec);
      if (score <= bestDot) continue;
      bestDot = score;
      out.valid = true;
      memcpy(out.vec, vec, DIM);
      for (size_t k = 0; k < WCLS_CLAUSE_TOKENS; ++k) {
        const uint16_t tok = readU16(slot + WCLS_TOKS_OFFSET + 2 * k);
        out.toks[k] = tok < PEWR_VOCAB ? tok : 0;
      }
      out.simQ7 = std::clamp<int32_t>(score >> 7, 0, 127);
      const std::string_view text =
          fixedText(slot + WCLS_TEXT_OFFSET, std::min<size_t>(textLen, WCLS_TEXT_MAX));
      memcpy(out.text, text.data(), text.size());
      out.text[text.size()] = '\0';
    }
  }
  return true;
}

bool readClausesHeader(HalFile& file) {
  uint8_t h[WCLS_HEADER_BYTES];
  return file.fileSize() == WCLS_BYTES && file.seekSet(0) && readFully(file, h, sizeof(h)) &&
         readU32(h) == WCLS_MAGIC && readU16(h + 4) == 1 && readU16(h + 6) == PEWR_BUCKETS &&
         readU16(h + 8) == PEWR_SLOTS && readU16(h + 10) == WCLS_SLOT_BYTES;
}

struct PeEntities {
  std::string_view subject, role, echo;
};

void runDecoder(const PeWriter& pe, const PeEntities& ents, PeWork& work) {
  bool used[TOK_FIRST_WORD] = {};
  uint8_t cursor[PE_PLAN_STEPS] = {};
  int8_t vHist[3][DIM] = {};  // v_{t-2}, v_{t-1}, v_t
  int32_t acc[DIM];
  int8_t u[DIM], c[DIM], h[DIM];
  uint16_t prev = TOK_BOS;
  std::fill(std::begin(work.tokens), std::end(work.tokens), 0);

  const auto selectable = [&](const uint16_t v) {
    if (v == TOK_BOS) return false;
    if (v >= TOK_FIRST_WORD || v == TOK_EOS) return true;
    if (used[v]) return false;
    if (v == TOK_SUBJECT) return !ents.subject.empty();
    if (v == TOK_ECHO) return !ents.echo.empty();
    if (v >= TOK_CLAUSE0) return work.clauses[v - TOK_CLAUSE0].valid;
    return true;  // <ENT_ROLE> falls back to "this concept"
  };

  for (size_t t = 0; t < PE_MAX_TOKENS; ++t) {
    const size_t s = std::min(PE_PLAN_STEPS - 1, t / PEWR_STEP_TOKENS);
    const PeClause& clause = work.clauses[s];
    const int8_t* emb = pe.at(PEWR_OFF_TOK + static_cast<size_t>(prev) * DIM);
    const int8_t* pos = pe.at(PEWR_OFF_POS + t * DIM);
    for (size_t d = 0; d < DIM; ++d) {
      const int32_t m = (work.plan[s][d] + clause.vec[d]) >> 1;
      u[d] = sat8((emb[d] + m + pos[d]) >> 1);
    }

    // Gated ShortConv (k = 3) over the gated inputs.
    pe.gate.apply(u, acc);
    memmove(vHist[0], vHist[1], 2 * DIM);
    for (size_t d = 0; d < DIM; ++d) vHist[2][d] = static_cast<int8_t>((u[d] * gateQ7(acc[d]) + 64) >> 7);
    for (size_t d = 0; d < DIM; ++d) {
      const int8_t* tap = pe.dw.w + d * 3;
      const int32_t sum = tap[0] * vHist[0][d] + tap[1] * vHist[1][d] + tap[2] * vHist[2][d];
      c[d] = sat8(sh(sum, pe.dw.s) + pe.dw.biasAt(d));
    }
    pe.outProj.apply(c, acc);
    for (size_t d = 0; d < DIM; ++d) h[d] = sat8(u[d] + acc[d]);

    // Dual-route head: vocabulary logits plus the clause copy boost.
    pe.lm.apply(h, work.logits);
    if (cursor[s] < WCLS_CLAUSE_TOKENS && clause.toks[cursor[s]] > 0) {
      int32_t gateLogit;
      pe.copy.apply(h, &gateLogit);
      work.logits[clause.toks[cursor[s]]] += (gateQ7(gateLogit) * clause.simQ7) >> 4;
    }
    const size_t from = t > PEWR_RECENT ? t - PEWR_RECENT : 0;
    for (size_t j = from; j < t; ++j) {
      const uint16_t v = work.tokens[j];
      if (v < TOK_FIRST_WORD || std::find(work.tokens + from, work.tokens + j, v) != work.tokens + j) continue;
      work.logits[v] -= PEWR_REPEAT_PENALTY;
    }

    int best = -1;
    for (size_t v = 0; v < PEWR_VOCAB; ++v) {
      if (!selectable(static_cast<uint16_t>(v))) continue;
      if (best < 0 || work.logits[v] > work.logits[best]) best = static_cast<int>(v);
    }
    const auto y = static_cast<uint16_t>(best);
    work.tokens[t] = y;
    if (y == TOK_EOS) break;
    if (y < TOK_FIRST_WORD) used[y] = true;
    if (y >= TOK_CLAUSE0 && y < TOK_FIRST_WORD) {
      cursor[y - TOK_CLAUSE0] = WCLS_CLAUSE_TOKENS;
    } else if (cursor[s] < WCLS_CLAUSE_TOKENS && clause.toks[cursor[s]] == y) {
      ++cursor[s];
    }
    prev = y;
  }
}

// The code point starting at text[i] spanning `n` bytes, or false when the
// lead byte is invalid, the sequence runs past the end, or a continuation
// byte is missing (no overlong or surrogate checks).
bool decodeAt(std::string_view text, const size_t i, uint32_t& cp, size_t& n) {
  const auto b = static_cast<uint8_t>(text[i]);
  if (b < 0x80) {
    n = 1;
    cp = b;
  } else if (b >= 0xC0 && b < 0xE0) {
    n = 2;
    cp = b & 0x1F;
  } else if (b >= 0xE0 && b < 0xF0) {
    n = 3;
    cp = b & 0x0F;
  } else if (b >= 0xF0 && b < 0xF8) {
    n = 4;
    cp = b & 0x07;
  } else {
    return false;
  }
  if (i + n > text.size()) return false;
  for (size_t k = 1; k < n; ++k) {
    const auto c = static_cast<uint8_t>(text[i + k]);
    if (!continuation(c)) return false;
    cp = (cp << 6) | (c & 0x3F);
  }
  return true;
}

bool cjkCodepoint(const uint32_t cp) {
  return (cp >= 0x2E80 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFF00 && cp <= 0xFFEF);
}

bool startsCjk(std::string_view text) {
  uint32_t cp;
  size_t n;
  return !text.empty() && decodeAt(text, 0, cp, n) && cjkCodepoint(cp);
}

// The last code point must decode from at most three continuation bytes back
// and end exactly at the end of `text`.
bool endsCjk(std::string_view text) {
  if (text.empty()) return false;
  size_t i = text.size() - 1;
  for (int k = 0; i > 0 && k < 3 && continuation(static_cast<uint8_t>(text[i])); ++k) --i;
  uint32_t cp;
  size_t n;
  return decodeAt(text, i, cp, n) && i + n == text.size() && cjkCodepoint(cp);
}

// Pieces join with one space, except: "##" pieces glue (the marker dropped),
// the first piece, a piece opening with , . ; : ! ? ) ', a piece after "(",
// and CJK after CJK. Cut at RIFF_MAX_BYTES on a code point boundary.
void renderRiff(const PeWriter& pe, const PeEntities& ents, const PeWork& work, std::string& text) {
  text.clear();
  for (const uint16_t token : work.tokens) {
    if (token == TOK_EOS) break;
    std::string_view piece;
    if (token >= TOK_FIRST_WORD) {
      piece = pe.vocab(token);
    } else if (token == TOK_SUBJECT) {
      piece = ents.subject;
    } else if (token == TOK_ROLE) {
      piece = ents.role.empty() ? std::string_view("this concept") : ents.role;
    } else if (token == TOK_ECHO) {
      piece = ents.echo;
    } else if (token >= TOK_CLAUSE0) {
      piece = work.clauses[token - TOK_CLAUSE0].text;
    }
    if (piece.empty()) continue;
    bool glue = piece.size() >= 2 && piece[0] == '#' && piece[1] == '#';
    if (glue) {
      piece.remove_prefix(2);
    } else {
      glue = text.empty() || strchr(",.;:!?)'", piece[0]) != nullptr || text.back() == '(' ||
             (endsCjk(text) && startsCjk(piece));
    }
    if (!glue) text += ' ';
    text.append(piece.data(), piece.size());
  }
  if (text.size() > RIFF_MAX_BYTES) {
    size_t n = RIFF_MAX_BYTES;
    while (n > 0 && continuation(static_cast<uint8_t>(text[n]))) --n;
    text.resize(n);
  }
}

std::string joinPath(const std::string& dir, const char* name) {
  if (dir.empty()) return std::string("/") + name;
  return dir.back() == '/' ? dir + name : dir + '/' + name;
}

constexpr size_t MARGIN_NOTE_MAX_BYTES = 480;

}  // namespace

uint64_t fnv1a64(std::string_view bytes) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (const char c : bytes) {
    hash ^= static_cast<uint8_t>(c);
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

int32_t dot(const int8_t a[DIM], const int8_t b[DIM]) {
  int32_t sum = 0;
  for (size_t d = 0; d < DIM; ++d) sum += static_cast<int32_t>(a[d]) * b[d];
  return sum;
}

void encodeFallback(std::string_view utf8Text, int8_t outVec[DIM]) {
  const size_t n = std::min(utf8Text.size(), MAX_INPUT_BYTES);
  int32_t acc[DIM] = {};
  uint8_t bytes[MAX_INPUT_BYTES];
  for (size_t i = 0; i < n; ++i) {
    const auto b = static_cast<uint8_t>(utf8Text[i]);
    bytes[i] = b >= 'A' && b <= 'Z' ? static_cast<uint8_t>(b + 32) : b;
  }
  const size_t gram = n < 3 ? n : 3;
  const size_t grams = n == 0 ? 0 : n - gram + 1;
  for (size_t i = 0; i < grams; ++i) {
    const uint64_t h = fnv1a64({reinterpret_cast<const char*>(bytes + i), gram});
    const uint64_t h2 = (h * 0x9E3779B97F4A7C15ULL) ^ (h >> 29);
    for (size_t d = 0; d < DIM; ++d) {
      const uint64_t bit = d < 64 ? (h >> d) & 1u : (h2 >> (d - 64)) & 1u;
      acc[d] += bit ? 1 : -1;
    }
  }
  normalize(acc, outVec);
}

bool encodeText(std::string_view utf8Text, int8_t outVec[DIM], const char* engramPath) {
  if (!loadModel()) {
    encodeFallback(utf8Text, outVec);
    return false;
  }
  const size_t n = std::min(utf8Text.size(), MAX_INPUT_BYTES);
  auto forward = makeUniqueNoThrow<Forward>();
  if (!forward) {
    LOG_ERR("CPT", "OOM: forward state");
    encodeFallback(utf8Text, outVec);
    return false;
  }
  forward->engramOpen = forward->engram.open(engramPath);
  forward->run(reinterpret_cast<const uint8_t*>(utf8Text.data()), n);
  if (forward->engramOpen) forward->engram.file.close();
  normalize(forward->sums, outVec);
  return true;
}

std::string bestDictionarySense(const int8_t contextVec[DIM], std::string_view plainDefinition) {
  const ParsedDefinition parsed = parseSenses(plainDefinition);
  const size_t best = bestSenseIndex(parsed, contextVec);
  if (best == std::string_view::npos) return {};
  return std::string(parsed.senses[best].text);
}

std::string rankDictionarySenses(const int8_t contextVec[DIM], std::string_view rawDefinition,
                                 std::string_view marker) {
  const ParsedDefinition parsed = parseSenses(rawDefinition);
  const size_t bestIndex = bestSenseIndex(parsed, contextVec);
  if (bestIndex == std::string_view::npos) return std::string(rawDefinition);
  const Sense& best = parsed.senses[bestIndex];
  const auto& lines = parsed.lines;

  std::string out;
  out.reserve(rawDefinition.size() + marker.size() + 64);
  for (size_t i = 0; i < parsed.headerEnd; ++i) {
    out.append(lines[i]);
    out += '\n';
  }
  out.append(marker);
  if (best.label != std::string_view::npos && best.label >= parsed.headerEnd) {
    out.append(trim(lines[best.label]));
    out += ' ';
  }
  out.append(best.text);
  for (size_t i = parsed.headerEnd; i < lines.size(); ++i) {
    if (i != best.line) {
      out += '\n';
      out.append(lines[i]);
      continue;
    }
    if (parsed.numbered) continue;
    // cedict: drop just this gloss from its "; " list.
    std::string rest;
    const auto segments = splitSegments(lines[i]);
    for (size_t s = 0; s < segments.size(); ++s) {
      if (s == best.segment || segments[s].empty()) continue;
      if (!rest.empty()) rest += "; ";
      rest.append(segments[s]);
    }
    if (!rest.empty()) {
      out += '\n';
      out += rest;
    }
  }
  return out;
}

std::string rankDictionarySenses(std::string_view contextSentence, std::string_view rawDefinition,
                                 std::string_view marker) {
  int8_t context[DIM];
  encodeText(contextSentence, context);
  return rankDictionarySenses(context, rawDefinition, marker);
}

std::string lookupEngramGloss(std::string_view word, const char* engramPath) {
  if (word.empty()) return {};
  Engram engram;
  if (!engram.open(engramPath)) return {};
  std::string lower(word);
  for (char& c : lower)
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
  std::string gloss;
  for (const std::string_view key : {word, std::string_view(lower)}) {
    const uint64_t hash = fnv1a64(key);
    uint8_t slot[ENGR_SLOT_BYTES];
    if (hash == 0 || !engram.readSlot(hash, slot) || readU64(slot) != hash) continue;
    const char* text = reinterpret_cast<const char*>(slot + ENGR_GLOSS_OFFSET);
    gloss.assign(text, strnlen(text, ENGR_GLOSS_BYTES));
    if (!gloss.empty()) break;
  }
  engram.file.close();
  return gloss;
}

bool appendEpisodicVector(std::string_view bookPath, const uint32_t pageIndex, const int8_t vec[DIM]) {
  if (bookPath.empty()) return false;
  HalFile file = Storage.open(paths.episodic, O_RDWR | O_CREAT);
  if (!file) return false;
  uint8_t header[ICPT_HEADER_BYTES];
  if (file.fileSize() < ICPT_HEADER_BYTES) {
    const uint32_t fields[4] = {ICPT_MAGIC, 1, 0, DIM};
    memcpy(header, fields, sizeof(header));
    if (!file.seekSet(0) || file.write(header, sizeof(header)) != sizeof(header)) {
      file.close();
      return false;
    }
  }
  const uint64_t pathHash = fnv1a64(bookPath);
  uint32_t count = 0;
  uint32_t same = UINT32_MAX;
  uint32_t oldest = 0;
  uint32_t oldestSeq = UINT32_MAX;
  uint32_t newestSeq = 0;
  const bool ok = forEachRecord(file, [&](const uint32_t index, const uint8_t* record) {
    count = index + 1;
    const uint32_t seq = readU32(record + 12);
    if (readU64(record) == pathHash && readU32(record + 8) == pageIndex) same = index;
    if (seq < oldestSeq) {
      oldestSeq = seq;
      oldest = index;
    }
    newestSeq = std::max(newestSeq, seq);
    return true;
  });
  if (!ok) {
    file.close();
    return false;
  }
  const uint32_t target = same != UINT32_MAX ? same : count < MAX_EPISODIC_RECORDS ? count : oldest;
  uint8_t record[ICPT_RECORD_BYTES] = {};
  const uint32_t seq = newestSeq + 1;
  memcpy(record, &pathHash, sizeof(pathHash));
  memcpy(record + 8, &pageIndex, sizeof(pageIndex));
  memcpy(record + 12, &seq, sizeof(seq));
  const std::string label = labelFor(bookPath);
  memcpy(record + ICPT_LABEL_OFFSET, label.data(), label.size());
  memcpy(record + ICPT_VEC_OFFSET, vec, DIM);
  bool written = file.seekSet(ICPT_HEADER_BYTES + static_cast<size_t>(target) * ICPT_RECORD_BYTES) &&
                 file.write(record, sizeof(record)) == sizeof(record);
  if (written && target == count) {
    const uint32_t newCount = count + 1;
    written = file.seekSet(8) && file.write(&newCount, sizeof(newCount)) == sizeof(newCount);
  }
  file.close();
  return written;
}

bool appendEpisodicPage(std::string_view bookPath, const uint32_t pageIndex, std::string_view pageText) {
  int8_t vec[DIM];
  encodeText(pageText, vec);
  return appendEpisodicVector(bookPath, pageIndex, vec);
}

std::vector<Match> queryEpisodicAndShelf(const int8_t queryVec[DIM], const size_t maxResults, const bool includeShelf) {
  std::vector<Match> top;
  for (const bool shelf : {false, true}) {
    if (shelf && !includeShelf) break;
    scanIndex(shelf ? paths.shelf : paths.episodic, [&](uint32_t, const uint8_t* record) {
      const int32_t score = recordDot(record, queryVec);
      if (top.size() < maxResults || score > top.back().dot) {
        Match match{readU64(record), readU32(record + 8), score, shelf, recordLabel(record)};
        memcpy(match.vec, record + ICPT_VEC_OFFSET, DIM);
        insertTop(top, std::move(match), maxResults);
      }
      return true;
    });
  }
  return top;
}

std::vector<Match> queryEpisodicAndShelf(std::string_view queryText, const size_t maxResults) {
  int8_t query[DIM];
  encodeText(queryText, query);
  return queryEpisodicAndShelf(query, maxResults);
}

bool latestEpisodicVector(int8_t outVec[DIM]) {
  bool found = false;
  uint32_t newest = 0;
  scanIndex(paths.episodic, [&](uint32_t, const uint8_t* record) {
    const uint32_t seq = readU32(record + 12);
    if (!found || seq >= newest) {
      newest = seq;
      memcpy(outVec, record + ICPT_VEC_OFFSET, DIM);
      found = true;
    }
    return true;
  });
  return found;
}

size_t scoreShelf(const int8_t queryVec[DIM], const uint64_t* pathHashes, const size_t count, int32_t* outDots) {
  std::fill(outDots, outDots + count, INT32_MIN);
  size_t matched = 0;
  scanIndex(paths.shelf, [&](uint32_t, const uint8_t* record) {
    const uint64_t hash = readU64(record);
    for (size_t i = 0; i < count; ++i) {
      if (pathHashes[i] != hash || outDots[i] != INT32_MIN) continue;
      outDots[i] = recordDot(record, queryVec);
      ++matched;
    }
    return matched < count;
  });
  return matched;
}

namespace {
int8_t currentContextVec[DIM];
bool hasCurrentContext = false;
}  // namespace

void setCurrentContext(const int8_t vec[DIM]) {
  memcpy(currentContextVec, vec, DIM);
  hasCurrentContext = true;
}

bool currentContext(int8_t outVec[DIM]) {
  if (!hasCurrentContext) return false;
  memcpy(outVec, currentContextVec, DIM);
  return true;
}

void clearCurrentContext() { hasCurrentContext = false; }

bool readingContext(int8_t outVec[DIM]) { return currentContext(outVec) || latestEpisodicVector(outVec); }

bool recordPageDwell(std::string_view bookPath, const uint32_t pageIndex, std::string_view slice,
                     const uint32_t dwellMs) {
  if (dwellMs < CONTEXT_DWELL_MS || slice.empty()) return false;
  int8_t vec[DIM];
  encodeText(slice, vec);
  setCurrentContext(vec);
  if (dwellMs >= EPISODE_DWELL_MS && !bookPath.empty()) appendEpisodicVector(bookPath, pageIndex, vec);
  return true;
}

std::string_view utf8Prefix(std::string_view text, const size_t maxBytes) {
  if (text.size() <= maxBytes) return text;
  size_t n = maxBytes;
  while (n > 0 && continuation(static_cast<uint8_t>(text[n]))) --n;
  return text.substr(0, n);
}

PageDwellQueue::Page& PageDwellQueue::frameShown(const uint32_t dwellMs, const uint32_t index, const uint32_t nowMs) {
  if (hasShown && dwellMs >= CONTEXT_DWELL_MS) {
    queued = shown;
    queued.dwellMs = dwellMs;
    hasQueued = true;
  }
  shown.index = index;
  shown.dwellMs = 0;
  shown.length = 0;
  shownAtMs = nowMs;
  hasShown = true;
  return shown;
}

bool PageDwellQueue::Page::append(std::string_view more) {
  const std::string_view part = utf8Prefix(more, DWELL_SLICE_BYTES - length);
  if (!part.empty()) memcpy(text + length, part.data(), part.size());
  length = static_cast<uint16_t>(length + part.size());
  return part.size() == more.size() && length < DWELL_SLICE_BYTES;
}

bool PageDwellQueue::take(Page& out, const uint32_t nowMs, const uint32_t settleMs) {
  if (!hasQueued || nowMs - shownAtMs < settleMs) return false;
  out = queued;
  hasQueued = false;
  return true;
}

std::string describeMatch(const Match& match) {
  if (match.shelf) return match.label;
  char position[32];
  const uint32_t chapter = match.page >> 16;
  if (chapter > 0) {
    snprintf(position, sizeof(position), " ch.%u p.%u", static_cast<unsigned>(chapter),
             static_cast<unsigned>((match.page & 0xFFFF) + 1));
  } else {
    snprintf(position, sizeof(position), " p.%u", static_cast<unsigned>(match.page + 1));
  }
  return match.label + position;
}

uint16_t cspwHalfLife(const uint8_t salience) {
  const uint32_t code = 64u + (salience >> 1);
  const uint32_t tau = (POW2_FRAC_Q15[code & 15] * (1u << (code >> 4)) + 16384u) >> 15;
  return static_cast<uint16_t>(std::max<uint32_t>(16, tau));
}

uint16_t cspwDecayQ15(const uint32_t deltaPages, const uint32_t tau) {
  if (tau == 0) return 0;
  const uint64_t x = (static_cast<uint64_t>(deltaPages) * 256u) / tau;  // half-lives in 1/256
  if (x >= 16 * 256) return 0;
  const auto octaves = static_cast<uint32_t>(x >> 8);
  const auto seg = static_cast<uint32_t>((x & 255) >> 4);
  const auto t = static_cast<uint32_t>(x & 15);
  const uint32_t base = (POW2_NEG_Q15[seg] * (16 - t) + POW2_NEG_Q15[seg + 1] * t + 8) >> 4;
  return static_cast<uint16_t>(octaves == 0 ? base : (base + (1u << (octaves - 1))) >> octaves);
}

bool runChronoSlots(const int8_t (*scenes)[DIM], const uint32_t* pages, const size_t count, const int8_t* echo,
                    ChronoSlots& out) {
  out = ChronoSlots{};
  if (!scenes || !pages || count == 0 || count > CSPW_MAX_SCENES || !loadCspw()) return false;
  auto st = makeUniqueNoThrow<CspwState>();
  if (!st) {
    LOG_ERR("CPT", "OOM: CS-PW state");
    return false;
  }
  for (size_t m = 0; m < count; ++m) {
    projectQ7<CSPW_D_MODEL>(scenes[m], DIM, cspwAt(CSPW_OFF_UP), st->h[m]);
    projectQ7<CSPW_D_MODEL>(st->h[m], CSPW_D_MODEL, cspwAt(CSPW_OFF_Q), st->q[m]);
    projectQ7<CSPW_KV>(st->h[m], CSPW_D_MODEL, cspwAt(CSPW_OFF_K), st->k[m]);
    projectQ7<CSPW_KV>(st->h[m], CSPW_D_MODEL, cspwAt(CSPW_OFF_V), st->v[m]);
  }
  const int8_t* salienceW = cspwAt(CSPW_OFF_SAL_W);
  const auto salienceB = static_cast<int16_t>(readU16(cspw.weights.get() + CSPW_OFF_SAL_B));
  for (size_t m = 0; m < count; ++m) {
    int8_t attn[CSPW_D_MODEL];
    chunkGqa(*st, m, attn);
    int32_t acc[CSPW_D_MODEL];
    matVec<CSPW_D_MODEL>(attn, cspwAt(CSPW_OFF_O), acc, CSPW_D_MODEL);
    for (size_t j = 0; j < CSPW_D_MODEL; ++j) st->h[m][j] = sat8(st->h[m][j] + (acc[j] >> 7));
    int8_t v128[DIM];
    projectQ7<DIM>(st->h[m], CSPW_D_MODEL, cspwAt(CSPW_OFF_DOWN), v128);
    for (size_t j = 0; j < DIM; ++j) st->r[m][j] = sat8((scenes[m][j] + v128[j]) >> 1);
    const int32_t salience = dot(salienceW, st->r[m]);
    out.salience[m] = static_cast<uint8_t>(std::clamp<int32_t>((salience >> 7) + salienceB + 128, 0, 255));
    out.tau[m] = cspwHalfLife(out.salience[m]);
    for (size_t k = 0; k < CSPW_SLOTS; ++k) {
      const int8_t* gate = cspwAt(CSPW_OFF_GATE + k * DIM * DIM);
      for (size_t d = 0; d < DIM; ++d) st->u[k][m][d] = sat8(dot(gate + d * DIM, st->r[m]) >> 7);
    }
  }

  // Slot 0 (Role): salience-weighted mean over every mention.
  int64_t den = 0;
  for (size_t m = 0; m < count; ++m) den += out.salience[m] + 32;
  for (size_t d = 0; d < DIM; ++d) {
    int64_t num = 0;
    for (size_t m = 0; m < count; ++m) num += static_cast<int64_t>(out.salience[m] + 32) * st->u[0][m][d];
    out.slot[0][d] = sat8(static_cast<int32_t>(floorDiv(2 * num + den, 2 * den)));
  }
  // Slot 1 (Origin): locked to the first scene.
  memcpy(out.slot[1], st->u[1][0], DIM);
  // Slot 2 (Recent Arc): CfC write gated by the page gap and salience.
  int8_t* arc = out.slot[2];
  memcpy(arc, st->u[2][0], DIM);
  for (size_t m = 1; m < count; ++m) {
    const uint32_t gap = pages[m] > pages[m - 1] ? pages[m] - pages[m - 1] : 0;
    const int32_t f = cspwDecayQ15(gap, out.tau[m]);
    const int32_t alpha = std::clamp<int32_t>((32768 - f) + ((out.salience[m] * f) >> 9), 4096, 32768);
    for (size_t d = 0; d < DIM; ++d) arc[d] = sat8(((32768 - alpha) * arc[d] + alpha * st->u[2][m][d] + 16384) >> 15);
  }
  // Slot 3 (Cross-Book Echo).
  const int8_t* last3 = st->u[3][count - 1];
  for (size_t d = 0; d < DIM; ++d) out.slot[3][d] = echo ? sat8((last3[d] + echo[d]) >> 1) : last3[d];

  const int8_t* frames = cspwAt(CSPW_OFF_FRAME_BANK);
  for (size_t k = 0; k < CSPW_SLOTS; ++k) {
    out.frame[k] = bestRow(frames, k * CSPW_FRAME_GROUP, CSPW_FRAME_GROUP, out.slot[k], frameText);
  }
  out.phrase = bestRow(cspwAt(CSPW_OFF_PHRASE_BANK), 0, CSPW_PHRASES, out.slot[0], phraseText);
  int32_t bestArc = INT32_MIN;
  for (size_t m = 1; m < count; ++m) {
    const int32_t score = dot(out.slot[2], st->u[2][m]);
    if (out.arcScene < 0 || score >= bestArc) {
      out.arcScene = static_cast<int16_t>(m);
      bestArc = score;
    }
  }
  return true;
}

bool weaveEntityRefresher(const char* word, const size_t wordLen, const char* bookPath, const uint32_t currentCutoff,
                          char* outStory, const size_t outStoryCap) {
  if (!outStory || outStoryCap == 0) return false;
  outStory[0] = '\0';
  if (!word || wordLen == 0) return false;
  StoryGraph graph;
  if (!graph.open(paths.storyGraph)) return false;

  // A later name of an entity only resolves once the story has revealed it.
  const std::string_view surface(word, wordLen);
  uint64_t term = fnv1a64(surface);
  uint64_t canonical = 0;
  if (graph.resolveAlias(term, currentCutoff, canonical)) term = canonical;
  uint8_t entity[SGRF_ENTITY_BYTES];
  const uint64_t book = bookPath && *bookPath ? fnv1a64(bookPath) : 0;
  if (!(book != 0 && graph.findEntity(term, book, entity)) && !graph.findEntity(term, 0, entity)) return false;
  // Hard spoiler gate: never name an entity before its first appearance.
  if (currentCutoff > 0 && readU32(entity + 16) > currentCutoff) return false;
  const uint32_t eventStart = readU32(entity + 20);
  const uint32_t eventCount = readU16(entity + 24);
  if (eventStart > graph.events || eventCount > graph.events - eventStart) return false;

  auto work = makeUniqueNoThrow<WeaveWork>();
  if (!work) {
    LOG_ERR("CPT", "OOM: story weave");
    return false;
  }
  ScenePick story;
  ScenePick cross;
  for (uint32_t first = 0; first < eventCount; first += SGRF_EVENT_BATCH) {
    const uint32_t n = std::min<uint32_t>(SGRF_EVENT_BATCH, eventCount - first);
    if (!graph.readAt(graph.eventsOffset + static_cast<size_t>(eventStart + first) * SGRF_EVENT_BYTES, work->batch,
                      n * SGRF_EVENT_BYTES)) {
      return false;
    }
    for (uint32_t i = 0; i < n; ++i) {
      const uint8_t* event = work->batch + i * SGRF_EVENT_BYTES;
      if (currentCutoff != 0 && readU32(event) > currentCutoff) continue;  // event spoiler gate
      (event[6] == SLOT_HINT_CROSS_BOOK ? cross : story).add(static_cast<int32_t>(first + i), event[7]);
    }
  }
  const bool hasEcho = story.first >= 0 && cross.first >= 0;
  int32_t picks[3];
  const size_t count = (story.first >= 0 ? story : cross).scenes(picks);
  if (count == 0) return false;
  for (size_t m = 0; m < count; ++m) {
    StoryScene& scene = work->scenes[m];
    if (!scene.load(graph, eventStart + static_cast<uint32_t>(picks[m]))) return false;
    memcpy(work->vecs[m], scene.vec(), DIM);
    work->pages[m] = scene.page;
  }
  const StoryScene& echo = work->echo;
  if (hasEcho &&
      !work->echo.load(graph, eventStart + static_cast<uint32_t>(cross.last >= 0 ? cross.last : cross.first))) {
    return false;
  }
  graph.file.close();

  ChronoSlots& slots = work->slots;
  if (!runChronoSlots(work->vecs, work->pages, count, hasEcho ? echo.vec() : nullptr, slots)) {
    slots.arcScene = count >= 2 ? static_cast<int16_t>(count - 1) : -1;
  }
  const StoryScene& origin = work->scenes[0];
  const StoryScene* arcScene = slots.arcScene >= 0 ? &work->scenes[slots.arcScene] : nullptr;
  FrameFields fields;
  fields.e = fixedText(entity + SGRF_NAME_OFFSET, SGRF_NAME_BYTES);
  if (fields.e.empty()) fields.e = surface;
  if (slots.phrase >= 0) fields.r = phraseText(static_cast<size_t>(slots.phrase));
  fields.o = origin.gloss;
  if (arcScene) fields.a = arcScene->gloss;
  if (hasEcho) fields.b = echo.target.empty() ? echo.gloss : echo.target;

  fields.l = work->scenes[count - 1].target;
  std::string text = "Story: " + weavePart(slots.frame[0], "%E", fields);
  fields.l = origin.target;
  std::string body = weavePart(slots.frame[1], "%O", fields);
  if (arcScene) {
    fields.l = arcScene->target.empty() ? origin.target : arcScene->target;
    const std::string recent = weavePart(slots.frame[2], "recently %A", fields);
    if (!recent.empty()) body += body.empty() ? recent : "; " + recent;
  }
  if (!body.empty()) text += " \xE2\x80\x94 " + body;
  if (text.back() != '.' && text.back() != '!' && text.back() != '?') text += '.';
  if (hasEcho) {
    fields.l = echo.target;
    const std::string echoPart = weavePart(slots.frame[3], "%B", fields);
    if (!echoPart.empty()) text += " | Echo: " + echoPart;
  }

  size_t n = std::min(text.size(), outStoryCap - 1);
  while (n > 0 && n < text.size() && continuation(static_cast<uint8_t>(text[n]))) --n;
  memcpy(outStory, text.data(), n);
  outStory[n] = '\0';
  return true;
}

bool generateMarginRiff(const std::string& modelsDir, const int8_t cPage[DIM], const int8_t cEcho[DIM],
                        const char* subject, const char* role, const char* echoTitle, std::string& outRiff,
                        int8_t outPlan[PE_PLAN_STEPS][DIM], uint8_t outBuckets[PE_PLAN_STEPS],
                        uint16_t outTokens[PE_MAX_TOKENS]) {
  outRiff.clear();
  if (!cPage) return false;
  const std::string writerPath = joinPath(modelsDir, PE_WRITER_FILE);
  const std::string clausesPath = joinPath(modelsDir, WRITER_CLAUSES_FILE);
  if (!Storage.exists(writerPath.c_str()) || !Storage.exists(clausesPath.c_str())) return false;

  HalFile clauses;
  if (!Storage.openFileForRead("PEW", clausesPath, clauses) || !readClausesHeader(clauses)) {
    LOG_ERR("PEW", "Bad writer-clauses.bin");
    return false;
  }
  // The weights live in PSRAM only for this call.
  HalMemory::PsramBuffer weights;
  {
    HalFile file;
    if (!Storage.openFileForRead("PEW", writerPath, file)) return false;
    if (file.fileSize() != PEWR_BYTES) {
      LOG_ERR("PEW", "pe-writer.bin size %u, expected %u", static_cast<unsigned>(file.fileSize()),
              static_cast<unsigned>(PEWR_BYTES));
      return false;
    }
    weights = HalMemory::allocatePsram(PEWR_BYTES);
    if (!weights) {
      LOG_ERR("PEW", "OOM: pe-writer weights");
      return false;
    }
    if (!readFully(file, weights.get(), PEWR_BYTES)) return false;
  }
  PeWriter pe;
  if (!pe.init(weights.get())) {
    LOG_ERR("PEW", "Bad pe-writer.bin header");
    return false;
  }
  auto work = makeUniqueNoThrow<PeWork>();
  if (!work) {
    LOG_ERR("PEW", "OOM: pe-writer work");
    return false;
  }

  static const int8_t noEcho[DIM] = {};
  runPlanner(pe, cPage, cEcho ? cEcho : noEcho, *work);
  for (size_t s = 0; s < PE_PLAN_STEPS; ++s) {
    work->buckets[s] = routeBucket(pe, work->plan[s]);
    if (!retrieveClause(clauses, work->buckets[s], work->plan[s], *work, work->clauses[s])) {
      LOG_ERR("PEW", "writer-clauses.bin read failed");
      return false;
    }
  }
  clauses.close();

  const PeEntities ents{subject ? subject : "", role ? role : "", echoTitle ? echoTitle : ""};
  runDecoder(pe, ents, *work);
  if (outPlan) memcpy(outPlan, work->plan, sizeof(work->plan));
  if (outBuckets) memcpy(outBuckets, work->buckets, sizeof(work->buckets));
  if (outTokens) memcpy(outTokens, work->tokens, sizeof(work->tokens));
  renderRiff(pe, ents, *work, outRiff);
  return !outRiff.empty();
}

bool appendMarginNote(std::string_view note, const char* path) {
  if (!path || !*path) return false;
  std::string line(utf8Prefix(note, MARGIN_NOTE_MAX_BYTES));
  for (char& ch : line) {
    if (ch == '\n' || ch == '\r') ch = ' ';
  }
  if (line.empty()) return false;

  // The newest notes already on the card (the file is newest last); a file
  // past the cap is only read from its tail.
  std::string existing;
  if (Storage.exists(path)) {
    HalFile in;
    if (Storage.openFileForRead("PEW", path, in)) {
      const size_t size = in.fileSize();
      const size_t cap = MARGIN_NOTES_MAX * (MARGIN_NOTE_MAX_BYTES + 1);
      const size_t skip = size > cap ? size - cap : 0;
      existing.resize(size - skip);
      if (!in.seekSet(skip) || !readFully(in, reinterpret_cast<uint8_t*>(existing.data()), existing.size())) {
        existing.clear();
      } else if (skip > 0) {
        const size_t newline = existing.find('\n');
        existing.erase(0, newline == std::string::npos ? existing.size() : newline + 1);
      }
    }
  }
  std::vector<std::string_view> kept;
  for (size_t start = 0; start < existing.size();) {
    size_t end = existing.find('\n', start);
    if (end == std::string::npos) end = existing.size();
    if (end > start) kept.emplace_back(existing.data() + start, end - start);
    start = end + 1;
  }
  const size_t first = kept.size() >= MARGIN_NOTES_MAX ? kept.size() - (MARGIN_NOTES_MAX - 1) : 0;

  const std::string temporary = std::string(path) + ".tmp";
  {
    HalFile out = Storage.open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
    if (!out) return false;
    bool ok = true;
    const auto writeLine = [&](const std::string_view text) {
      ok = ok && out.write(text.data(), text.size()) == text.size() && out.write("\n", 1) == 1;
    };
    for (size_t i = first; i < kept.size(); ++i) writeLine(kept[i]);
    writeLine(line);
    out.close();
    if (!ok) {
      Storage.remove(temporary.c_str());
      return false;
    }
  }
  if (Storage.exists(path) && !Storage.remove(path)) {
    Storage.remove(temporary.c_str());
    return false;
  }
  return Storage.rename(temporary.c_str(), path);
}

void setPaths(const Paths& newPaths) {
  if (!paths.model || !newPaths.model || strcmp(paths.model, newPaths.model) != 0) unloadModel();
  if (!paths.cspw || !newPaths.cspw || strcmp(paths.cspw, newPaths.cspw) != 0) cspw = Cspw{};
  paths = newPaths;
}

void unloadModel() {
  model = Model{};
  cspw = Cspw{};
}

}  // namespace ConceptEngine
