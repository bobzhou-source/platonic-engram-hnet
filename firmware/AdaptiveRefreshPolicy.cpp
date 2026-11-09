#include "AdaptiveRefreshPolicy.h"

#include <string.h>

namespace {

constexpr uint32_t kMaxU32 = 0xFFFFFFFFu;

uint32_t saturate(uint64_t v) { return v > kMaxU32 ? kMaxU32 : static_cast<uint32_t>(v); }

// Sparse side of a frame: minority polarity pixels. Ink above P is a caller
// bug; clamp it rather than wrap.
uint64_t sparseSide(uint64_t ink, uint64_t pixels) {
  if (ink > pixels) ink = pixels;
  const uint64_t rest = pixels - ink;
  return ink < rest ? ink : rest;
}

bool inkMajority(uint64_t ink, uint64_t pixels) { return ink * 2 > pixels; }

// --- LiquidRefresh fixed point (32-bit only) --------------------------------
constexpr uint32_t kQ16 = 65536;
constexpr int32_t kRatioCap = 30000;  // readout features fit int16

constexpr int32_t clampI(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : v > hi ? hi : v; }

// a * p / 1000 rounded down, exact, without 64-bit math (p <= 1000).
uint32_t permilleOf(uint32_t a, uint32_t p) { return a / 1000 * p + a % 1000 * p / 1000; }

// exp(-y/1000) in Q16 (0 beyond y = 16), table-free: halve x = y/1000 until
// x <= 0.5, take the Pade(2,2) rational (12 - 6x + x^2) / (12 + 6x + x^2),
// square back.
constexpr uint32_t expNegQ16(uint32_t yMilli) {
  if (yMilli >= 16000) return 0;
  uint32_t x = yMilli * 8389u >> 7;  // y/1000 in Q16 (65.536 ~ 8389/128)
  unsigned k = 0;
  while (x > kQ16 / 2) {
    x >>= 1;
    ++k;
  }
  const uint32_t x2 = x * x >> 16;
  uint32_t r = ((12 * kQ16 - 6 * x + x2) << 12) / ((12 * kQ16 + 6 * x + x2) >> 4);
  if (r > 0xFFFF) r = 0xFFFF;
  while (k--) r = r * r >> 16;
  return r;
}

// softplus(x/1000) in Q16: max(x, 0) + ln(1 + e^-|x|), with
// ln(1 + u) = 2 atanh(s), s = u / (2 + u) <= 1/3: 2 (s + s^3/3 + s^5/5).
// Only evaluated at compile time (kRateQ16).
constexpr uint32_t softplusQ16(int32_t xMilli) {
  const uint32_t u = expNegQ16(static_cast<uint32_t>(xMilli < 0 ? -xMilli : xMilli));
  const uint32_t s = (u << 16) / (2 * kQ16 + u);
  const uint32_t s2 = s * s >> 16;
  const uint32_t s3 = s * s2 >> 16;
  return (xMilli > 0 ? static_cast<uint32_t>(xMilli) * 8389u >> 7 : 0) + 2 * (s + s3 / 3 + (s3 * s2 >> 16) / 5);
}

// Relaxation rate softplus(f(polarity, class)) per second, Q16.
constexpr uint32_t rateQ16(bool inverted, int cls) {
  return softplusQ16(AdaptiveRefreshPolicy::kLiquidLogitMilli[cls] +
                     (inverted ? AdaptiveRefreshPolicy::kLiquidLogitInvertedMilli : 0));
}
constexpr uint16_t kRateQ16[2][3] = {{rateQ16(false, 0), rateQ16(false, 1), rateQ16(false, 2)},
                                     {rateQ16(true, 0), rateQ16(true, 1), rateQ16(true, 2)}};
// elapsedMs is clamped to kMaxDwellMs so elapsedMs * rate fits 32 bits; the
// slowest rate has relaxed to 0 permille well before it.
constexpr uint32_t kMaxDwellMs = 1u << 20;
static_assert(kRateQ16[0][0] < 4096 && expNegQ16(kMaxDwellMs * kRateQ16[1][2] >> 16) * 1000 < kQ16 / 2,
              "dwell clamp must not change rho");

int16_t addSat16(int32_t a, int32_t b) { return static_cast<int16_t>(clampI(a + b, -32768, 32767)); }

// --- LiquidRefresh CfC v2 ----------------------------------------------------
// BEGIN CFC V2 WEIGHTS (generated offline by fit_liquid_cfc_v2.py; do not edit by hand)
// seed 20260517, 2400 float + 800 QAT iterations, 19 usage sessions; 339 bytes
// Rows: hidden units 0-5 text charge, 6-11 image sticking, 12-15 free. Columns: changed, fast turn,
// image area, dither, overlap, inverted, edge, mid-gray (Q7). Real value = int8 / 16.
constexpr int8_t kCfcWf[16][8] = {
    {0, 91, 1, -2, 0, 0, -1, -15},
    {-4, 90, 3, 0, 0, 0, -14, -8},
    {0, 91, 7, -8, 0, 0, 35, -27},
    {0, 91, -1, -3, 0, 0, -12, 1},
    {-1, 91, 0, 0, 0, 0, -2, -13},
    {-10, 91, 28, -11, 1, 0, -18, 27},
    {-20, -2, 3, 21, -2, 0, -11, -24},
    {-23, -3, 4, 20, -2, 0, 6, -9},
    {-23, -3, 8, 19, -2, 0, 78, 1},
    {-23, -3, 4, 22, -2, 0, 20, -9},
    {-23, -3, 4, 21, -2, 0, 2, -17},
    {-30, -3, 18, 9, -2, -1, 50, -26},
    {-65, 54, 115, 25, -127, -5, 0, 0},
    {-3, 26, -11, 0, 35, -10, -3, -8},
    {85, 90, -3, 15, 37, 80, -17, -10},
    {18, -2, -9, 36, 111, 12, 21, -3}};
constexpr int8_t kCfcBf[16] = {-47, -46, -47, -47, -47, -44, -50, -49, -49, -49, -49, -46, 3, -32, -8, 2};
constexpr int8_t kCfcWg[16][8] = {
    {-4, 60, 9, -5, -1, 0, 6, -5},
    {-1, 60, 2, -1, -1, 0, 21, 10},
    {-1, 60, 3, -2, -1, 0, -20, -7},
    {-2, 60, 3, 2, -1, 0, 4, -2},
    {-1, 60, -1, 0, -1, 0, -4, -4},
    {-28, 64, 16, 0, -1, -1, -41, 35},
    {-7, -1, 13, 4, 0, 0, 9, -4},
    {-10, -1, 14, 1, 0, 0, 12, -1},
    {-4, -1, 10, 6, 1, 0, -34, 18},
    {-10, -1, 6, 4, -1, 0, -3, -12},
    {-9, -1, 8, 3, 0, 0, 7, -13},
    {-7, -1, 15, 4, 1, 0, -45, -27},
    {-127, -50, 39, 7, -23, 0, -27, 8},
    {-6, -12, 2, 7, 61, -1, 3, 2},
    {-46, -2, 6, 0, 57, -4, 0, 1},
    {-40, -3, -18, 7, -57, -3, -4, 2}};
constexpr int8_t kCfcBg[16] = {-10, -11, -11, -11, -11, -4, 0, 1, -1, 2, 1, 0, -10, -2, 0, -1};
// Readouts: y0 refresh gate (fitted; the device learns its own copy), y1 contrast gain, y2 error bleed.
constexpr int8_t kCfcWout[3][16] = {
    {14, -22, -12, -5, -5, 29, 12, 40, -7, 5, 16, -20, 127, -27, 50, -61},
    {1, 3, 0, 2, -6, -2, 3, 0, 1, 0, 4, -3, -9, -1, 0, -6},
    {1, -5, 3, -1, 1, 8, -2, -1, 5, -1, -3, 8, -9, 0, 0, -1}};
constexpr int8_t kCfcBout[3] = {-12, 3, -1};
// END CFC V2 WEIGHTS
static_assert(sizeof kCfcWf + sizeof kCfcBf + sizeof kCfcWg + sizeof kCfcBg + sizeof kCfcWout + sizeof kCfcBout == 339,
              "CfC v2 parameters: 2 x (16x8 + 16) + 3x16 + 3 int8");

constexpr uint64_t kAllTiles = (uint64_t{1} << AdaptiveRefreshPolicy::kTileCount) - 1;

// Permille (0..1000) as a Q7 input (0..128), rounded.
uint8_t q7FromPermille(uint32_t permille) {
  if (permille > 1000) permille = 1000;
  return static_cast<uint8_t>((permille * 128 + 500) / 1000);
}

// W x + 128 b: the pre-activation in 1/2048 (weights 1/16, inputs 1/128).
int32_t cfcPreact(const int8_t w[AdaptiveRefreshPolicy::kCfcInputs], const int8_t b,
                  const uint8_t x[AdaptiveRefreshPolicy::kCfcInputs]) {
  int32_t a = 128 * static_cast<int32_t>(b);
  for (uint8_t i = 0; i < AdaptiveRefreshPolicy::kCfcInputs; ++i) a += static_cast<int32_t>(w[i]) * x[i];
  return a;
}

// 1/2048 -> thousandths, truncated toward zero.
int32_t cfcMilli(const int32_t a) { return a * 125 / 256; }

// exp(-softplus(f) * dt) in Q16; exactly 1.0 when nothing elapses.
uint32_t cfcRhoQ16(const int32_t fPreact, const uint32_t dtMs) {
  const uint64_t y = static_cast<uint64_t>(softplusQ16(cfcMilli(fPreact))) * dtMs >> 16;  // exponent, thousandths
  if (y == 0) return kQ16;
  return y >= 16000 ? 0 : expNegQ16(static_cast<uint32_t>(y));
}

// tanh(m / 1000) in Q8 via (1 - e^-2|m|) / (1 + e^-2|m|), rounded.
int16_t cfcTanhQ8(const int32_t milli) {
  const uint32_t a = static_cast<uint32_t>(milli < 0 ? -milli : milli);
  const uint32_t u = a >= 8000 ? 0 : expNegQ16(2 * a);
  const int32_t t = static_cast<int32_t>((((kQ16 - u) << 8) + (kQ16 + u) / 2) / (kQ16 + u));
  return static_cast<int16_t>(milli < 0 ? -t : t);
}

// One closed-form step of every unit over dtMs:
//   h' = rho h + (1 - rho) g,  rho = exp(-softplus(W_f x + b_f) dt),  g = tanh(W_g x + b_g).
void cfcStep(const int16_t h[AdaptiveRefreshPolicy::kCfcUnits], const uint8_t x[AdaptiveRefreshPolicy::kCfcInputs],
             uint32_t dtMs, int16_t out[AdaptiveRefreshPolicy::kCfcUnits]) {
  if (dtMs > AdaptiveRefreshPolicy::kCfcMaxStepMs) dtMs = AdaptiveRefreshPolicy::kCfcMaxStepMs;
  for (uint8_t j = 0; j < AdaptiveRefreshPolicy::kCfcUnits; ++j) {
    const int64_t rho = cfcRhoQ16(cfcPreact(kCfcWf[j], kCfcBf[j], x), dtMs);
    const int64_t g = cfcTanhQ8(cfcMilli(cfcPreact(kCfcWg[j], kCfcBg[j], x)));
    out[j] = static_cast<int16_t>((rho * h[j] + (static_cast<int64_t>(kQ16) - rho) * g) >> 16);
  }
}

// W h + 256 b: a readout in 1/4096 (weights 1/16, h Q8).
int32_t cfcReadoutOf(const int8_t w[AdaptiveRefreshPolicy::kCfcUnits], const int32_t b,
                     const int16_t h[AdaptiveRefreshPolicy::kCfcUnits]) {
  int32_t y = 256 * b;
  for (uint8_t j = 0; j < AdaptiveRefreshPolicy::kCfcUnits; ++j) y += static_cast<int32_t>(w[j]) * h[j];
  return y;
}

// LQRF on disk (/.crosspoint/liquid-refresh.bin), little-endian. v1 is the
// first 28 bytes plus its check; v2 appends the CfC y0 readout.
struct LearnedFileV1 {
  uint32_t magic;
  uint16_t version;
  uint16_t bytes;
  int16_t w[4];  // v0, v1, v2, b
  uint32_t updates;
  uint8_t reserved[8];
  uint32_t check;  // FNV-1a 32 of the 28 bytes before it
};
struct LearnedFile {
  uint32_t magic;
  uint16_t version;
  uint16_t bytes;
  int16_t w[4];  // v0, v1, v2, b
  uint32_t updates;
  uint8_t reserved[8];
  int8_t cfcW0[AdaptiveRefreshPolicy::kCfcUnits];  // CfC y0 readout, 1/16
  int16_t cfcB0;
  uint16_t reserved2;
  uint32_t cfcUpdates;
  uint8_t reserved3[8];
  uint32_t check;  // FNV-1a 32 of the 60 bytes before it
};
static_assert(sizeof(LearnedFileV1) == AdaptiveRefreshPolicy::kLearnedStateV1Bytes, "LQRF v1 layout");
static_assert(sizeof(LearnedFile) == AdaptiveRefreshPolicy::kLearnedStateBytes, "LQRF v2 layout");
static_assert(offsetof(LearnedFile, cfcW0) == offsetof(LearnedFileV1, check), "LQRF v2 extends v1");
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "LQRF is stored as the in-memory struct");

uint32_t fnv1a32(const uint8_t* p, size_t n) {
  uint32_t h = 2166136261u;
  while (n--) h = (h ^ *p++) * 16777619u;
  return h;
}

constexpr AdaptiveRefreshPolicy::CfcReadout fittedReadout() {
  AdaptiveRefreshPolicy::CfcReadout r{};
  for (uint8_t j = 0; j < AdaptiveRefreshPolicy::kCfcUnits; ++j) r.w[j] = kCfcWout[0][j];
  r.b = kCfcBout[0];
  return r;
}
constexpr AdaptiveRefreshPolicy::CfcReadout kCfcFitted = fittedReadout();

}  // namespace

const AdaptiveRefreshPolicy::CfcReadout& AdaptiveRefreshPolicy::cfcFittedReadout() { return kCfcFitted; }

AdaptiveRefreshPolicy::CfcReadout AdaptiveRefreshPolicy::cfcReadout() const {
  CfcReadout r{};
  memcpy(r.w, cfcReadoutW0_, sizeof r.w);
  r.b = cfcReadoutB0_;
  return r;
}

int16_t AdaptiveRefreshPolicy::cfcGroupMean(const uint8_t first) const {
  int32_t sum = 0;
  for (uint8_t j = first; j < first + kCfcGroupUnits; ++j) sum += cfcHiddenQ8_[j];
  return static_cast<int16_t>(sum / kCfcGroupUnits);
}

void AdaptiveRefreshPolicy::resetCfcEpisode() {
  memset(cfcHiddenQ8_, 0, sizeof cfcHiddenQ8_);
  memset(cfcPendingQ8_, 0, sizeof cfcPendingQ8_);
  memset(cfcTraceQ8_, 0, sizeof cfcTraceQ8_);
  cfcTraceBiasQ8_ = 0;
  cfcTraceTurns_ = 0;
  lastImageTileMask_ = 0;
  cfcImageAreaPermille_ = 0;
  cfcExposedMs_ = 0;
  cfcLogit_ = 0;
  cfcInverted_ = false;
}

void AdaptiveRefreshPolicy::reset(bool initialClearRequired) {
  static_assert(sizeof(Liquid) + sizeof(LiquidReadout) <= 232, "LiquidRefresh RAM budget");
  debt_ = 0;
  pendingCharge_ = 0;
  pendingInk_ = 0;
  prevInk_ = 0;
  clearedCharge_ = 0;
  fastSinceClear_ = 0;
  initialClear_ = initialClearRequired;
  clearRequested_ = false;
  pendingValid_ = false;
  pendingFrameValid_ = false;
  prevInkKnown_ = false;
  pendingRecord_ = ClearRecord{};
  clearSource_ = ClearSource::None;
  imagePages_ = 0;
  grayPages_ = 0;
  imageRegion_ = Rect{};
  imageRegionUnknown_ = false;
  const uint32_t updates = liquid_.updates;
  liquid_ = Liquid{};
  liquid_.updates = updates;
  liquid_.retention = 1000;
  liquid_.tileGain = 1000;
  resetCfcEpisode();
}

uint32_t AdaptiveRefreshPolicy::estimateChangedPixels(const freeink::FrameChangeMetrics& frame) const {
  const uint64_t pixels = frame.pixelCount;
  const uint64_t curInk = frame.inkPixels;
  const uint64_t prevInk = prevInkKnown_ ? prevInk_ : curInk;
  const uint64_t sum = sparseSide(curInk, pixels) + sparseSide(prevInk, pixels);
  const uint64_t bounded = sum < pixels ? sum : pixels;
  const bool flipped = inkMajority(curInk > pixels ? pixels : curInk, pixels) !=
                       inkMajority(prevInk > pixels ? pixels : prevInk, pixels);
  return static_cast<uint32_t>(flipped ? pixels - bounded : bounded);
}

uint32_t AdaptiveRefreshPolicy::weightFor(const Inputs& in) const {
  uint64_t weight = kWeightUnitPermille;
  if (in.visibleGrayscale) weight += config_.weightGrayscalePermille;
  if (in.hasImage) weight += config_.weightImagePermille;
  if (in.visibleGrayscale && in.hasImage) weight += config_.weightGrayscaleImageExtraPermille;
  // Fast turn: the panel had no time to settle, charge up to the boost more.
  const uint32_t dt = elapsedFor(in);
  if (dt != 0 && dt < kFastTurnMs) {
    weight = weight * (1000 + kFastTurnBoostPermille * (kFastTurnMs - dt) / kFastTurnMs) / 1000;
  }
  return saturate(weight);
}

uint32_t AdaptiveRefreshPolicy::chargeFor(const Inputs& in) const {
  const auto& f = in.frame;
  uint64_t changed = f.historyValid ? f.changedPixels : estimateChangedPixels(f);
  if (changed > f.pixelCount) changed = f.pixelCount;
  const uint64_t flat = static_cast<uint64_t>(f.pixelCount) * config_.frameChargePermille / 1000;
  return saturate(changed * weightFor(in) / kWeightUnitPermille + flat);
}

uint16_t AdaptiveRefreshPolicy::liquidRhoPermille(uint32_t elapsedMs, const uint8_t cls, const bool inverted) {
  if (elapsedMs > kMaxDwellMs) elapsedMs = kMaxDwellMs;
  const uint32_t rate = kRateQ16[inverted][cls < 3 ? cls : 2];
  return static_cast<uint16_t>((expNegQ16(elapsedMs * rate >> 16) * 1000 + kQ16 / 2) >> 16);
}

int32_t AdaptiveRefreshPolicy::tileNext(uint8_t tile) const {
  const uint32_t v = static_cast<uint32_t>(liquid_.tile[tile]) * liquid_.retention / 1000 +
                     liquid_.staged[tile] * 4u * liquid_.tileGain / 1000;
  return v > 32767 ? 32767 : static_cast<int32_t>(v);
}

// Prepares this frame's liquid step and answers the debt rule. Pure with
// respect to committed state, like decide().
bool AdaptiveRefreshPolicy::liquidDebtDue(const Inputs& in) {
  const auto& f = in.frame;
  const uint32_t threshold = in.invertedOutput ? config_.nightClearThresholdPermille : config_.clearThresholdPermille;
  const uint32_t dt = elapsedFor(in);
  const uint32_t rho = liquidRhoPermille(dt, liquid_.residentClass, in.invertedOutput);
  const uint32_t floor = liquid_.debtFloor < debt_ ? liquid_.debtFloor : debt_;
  liquid_.pendingDebt = floor + permilleOf(debt_ - floor, rho);
  liquid_.retention = static_cast<uint16_t>(kLiquidFloorPermille + (1000 - kLiquidFloorPermille) * rho / 1000);
  const uint32_t gain = weightFor(in);
  liquid_.tileGain = static_cast<uint16_t>(gain > 0xFFFF ? 0xFFFF : gain);
  liquid_.pendingClass = in.hasImage ? kLiquidImage : in.visibleGrayscale ? kLiquidGray : kLiquidText;
  uint8_t streak = liquid_.fastStreak;
  if (dt >= kFastTurnMs) {
    streak = 0;
  } else if (dt != 0 && streak < kFastStreakFull) {
    ++streak;
  }
  liquid_.pendingStreak = streak;

  const uint64_t debt = static_cast<uint64_t>(liquid_.pendingDebt) + pendingCharge_;
  const uint64_t budget = static_cast<uint64_t>(f.pixelCount) * threshold / 1000;
  const bool tiles = liquid_.tilesActive || liquid_.tilesStaged;
  int16_t* const psi = liquid_.psi;
  uint32_t d = saturate(debt);
  uint32_t b = saturate(budget);
  while (d > kMaxU32 / 1000) {  // keep d * 1000 in 32 bits
    d >>= 1;
    b >>= 1;
  }
  psi[kReadoutDebt] = static_cast<int16_t>(d / 30 >= b ? kRatioCap : d * 1000 / b);
  psi[kReadoutPeak] = 0;
  if (tiles && threshold != 0) {
    int32_t peak = 0;
    int32_t sum = 0;
    for (uint8_t i = 0; i < kTileCount; ++i) {
      const int32_t v = tileNext(i);
      if (v > peak) peak = v;
      sum += v;
    }
    const uint32_t excess = static_cast<uint32_t>(peak - sum / kTileCount) * 1000 / (threshold * kTilePeakScale);
    psi[kReadoutPeak] = static_cast<int16_t>(excess > kRatioCap ? kRatioCap : excess);
  }
  psi[kReadoutStreak] = static_cast<int16_t>(streak * 1000 / kFastStreakFull);
  psi[kReadoutBias] = 1000;
  int32_t z = 0;
  for (int i = 0; i < 4; ++i) z += readout_.w[i] * psi[i];
  z /= 1000;
  liquid_.pressure = static_cast<int16_t>(clampI(z, 0, 1000));
  // Default readout, no tiles, no streak: z >= 1000 is exactly debt >= budget;
  // compare the integers themselves so the content-v2 rule is bit-exact.
  if (!tiles && streak == 0 && readout_ == LiquidReadout{}) return debt >= budget;
  return z >= 1000;
}

AdaptiveRefreshPolicy::Decision AdaptiveRefreshPolicy::decide(const Inputs& in) {
  const auto& f = in.frame;
  pendingValid_ = true;
  pendingFrameValid_ = f.frameValid && f.pixelCount != 0;
  pendingInk_ = f.inkPixels;
  pendingCharge_ = pendingFrameValid_ ? chargeFor(in) : 0;
  const bool debtDue = liquidDebtDue(in);
  const bool sticking = cfcDecide(in);

  Reason reason = Reason::None;
  if (initialClear_) {
    reason = Reason::Initial;
  } else if (clearRequested_) {
    reason = Reason::Requested;
  } else if (in.externalClearPending) {
    reason = Reason::External;
  } else if (!pendingFrameValid_) {
    reason = Reason::InvalidFrame;
  } else if (f.polarityChangePending) {
    reason = Reason::PolarityChange;
  } else if (debtDue) {
    reason = Reason::Debt;
  } else if (sticking) {
    reason = Reason::ImageSticking;
  } else if (fastSinceClear_ >= config_.safetyCap) {
    reason = Reason::SafetyCap;
  }
  pendingRecord_ = labelFor(in, reason);
  if (reason == Reason::Requested && clearSource_ == ClearSource::Image && config_.imageClearRegion) {
    Rect region{};
    pendingRecord_.regionFallback = regionFor(in, debtDue, region);
    if (pendingRecord_.regionFallback == RegionFallback::None) {
      pendingRecord_.regionOnly = true;
      pendingRecord_.region = region;
      return {Refresh::Region, reason, region};
    }
  }
  return {reason == Reason::None ? Refresh::Fast : Refresh::Clear, reason};
}

AdaptiveRefreshPolicy::RegionFallback AdaptiveRefreshPolicy::regionFor(const Inputs& in, const bool debtDue,
                                                                       Rect& region) const {
  const auto& f = in.frame;
  // The rest of decide()'s chain: any of these needs the whole screen.
  if (in.externalClearPending || !pendingFrameValid_ || f.polarityChangePending || debtDue ||
      fastSinceClear_ >= config_.safetyCap) {
    return RegionFallback::Overruled;
  }
  if (imageRegionUnknown_) return RegionFallback::Unknown;
  const int32_t x0 = imageRegion_.x > 0 ? imageRegion_.x : 0;
  const int32_t y0 = imageRegion_.y > 0 ? imageRegion_.y : 0;
  const int32_t right = static_cast<int32_t>(imageRegion_.x) + imageRegion_.w;
  const int32_t bottom = static_cast<int32_t>(imageRegion_.y) + imageRegion_.h;
  const int32_t x1 = right < in.screenWidth ? right : in.screenWidth;
  const int32_t y1 = bottom < in.screenHeight ? bottom : in.screenHeight;
  if (imageRegion_.empty() || x1 <= x0 || y1 <= y0) return RegionFallback::OffScreen;
  const uint64_t area = static_cast<uint64_t>(x1 - x0) * static_cast<uint64_t>(y1 - y0);
  const uint64_t screen = static_cast<uint64_t>(in.screenWidth) * in.screenHeight;
  if (area * 1000 > screen * kRegionMaxPermille) return RegionFallback::TooLarge;
  if (!in.regionClearAvailable) return RegionFallback::Unavailable;
  region = {static_cast<int16_t>(x0), static_cast<int16_t>(y0), static_cast<int16_t>(x1 - x0),
            static_cast<int16_t>(y1 - y0)};
  return RegionFallback::None;
}

AdaptiveRefreshPolicy::ClearRecord AdaptiveRefreshPolicy::labelFor(const Inputs& in, const Reason reason) const {
  ClearRecord record;
  record.reason = reason;
  if (reason == Reason::Requested) {
    record.source = clearSource_ == ClearSource::None ? ClearSource::Other : clearSource_;
  } else if (reason == Reason::External) {
    record.source = ClearSource::External;
  }
  if ((reason == Reason::Requested || reason == Reason::External) && in.externalClearPending && in.externalManual) {
    record.source = ClearSource::Manual;
  }
  const uint64_t pixels = pendingFrameValid_ ? in.frame.pixelCount : 0;
  if (pixels) {
    record.debtPermille = saturate((static_cast<uint64_t>(liquid_.pendingDebt) + pendingCharge_) * 1000 / pixels);
  }
  record.thresholdPermille = in.invertedOutput ? config_.nightClearThresholdPermille : config_.clearThresholdPermille;
  record.inverted = in.invertedOutput;
  record.image = in.hasImage;
  record.grayscale = in.visibleGrayscale;
  return record;
}

void AdaptiveRefreshPolicy::commit(Refresh displayed) {
  cfcCommit(displayed);
  liquidCommit(displayed);
  if (displayed == Refresh::Clear) {
    if (observer_) {
      ClearRecord record = pendingValid_ ? pendingRecord_ : ClearRecord{};
      if (!pendingValid_) record.source = clearSource_;
      record.fastTurns = fastSinceClear_;
      record.imagePages = imagePages_;
      record.grayPages = grayPages_;
      observer_(observerContext_, record);
    }
    debt_ = 0;
    fastSinceClear_ = 0;
    initialClear_ = false;
    clearRequested_ = false;
    clearedCharge_ = pendingValid_ ? pendingCharge_ : 0;
    clearSource_ = ClearSource::None;
    imagePages_ = 0;
    grayPages_ = 0;
    imageRegion_ = Rect{};
    imageRegionUnknown_ = false;
  } else {
    if (displayed == Refresh::Region) {
      // Ends the picture clear only; the frame itself was drawn fast and the
      // debt window carries on.
      if (observer_) {
        ClearRecord record = pendingValid_ ? pendingRecord_ : ClearRecord{};
        if (!pendingValid_) record.source = clearSource_;
        record.regionOnly = true;
        record.fastTurns = fastSinceClear_;
        record.imagePages = imagePages_;
        record.grayPages = grayPages_;
        observer_(observerContext_, record);
      }
      clearRequested_ = false;
      clearSource_ = ClearSource::None;
      imageRegion_ = Rect{};
      imageRegionUnknown_ = false;
    }
    clearedCharge_ = 0;
    if (pendingValid_) {
      debt_ = saturate(static_cast<uint64_t>(liquid_.pendingDebt) + pendingCharge_);
      if (fastSinceClear_ != 0xFFFFu) ++fastSinceClear_;
      if (pendingRecord_.image && imagePages_ != 0xFFFFu) ++imagePages_;
      if (pendingRecord_.grayscale && grayPages_ != 0xFFFFu) ++grayPages_;
    }
  }
  if (pendingValid_ && pendingFrameValid_) {
    prevInk_ = pendingInk_;
    prevInkKnown_ = true;
  } else {
    prevInkKnown_ = false;
  }
  pendingValid_ = false;
}

// Runs before commit() books the debt: reads the pending decide() state.
void AdaptiveRefreshPolicy::liquidCommit(const Refresh displayed) {
  if (pendingValid_) {
    liquid_.fastStreak = liquid_.pendingStreak;
    if (pendingFrameValid_) liquid_.residentClass = liquid_.pendingClass;
  }
  if (displayed == Refresh::Clear) {
    for (auto& t : liquid_.tile) t = 0;
    liquid_.tilesActive = false;
    liquid_.debtFloor = 0;
    // A manual clear keeps the trace for recordManualRefresh(); any other
    // clear ends the episode without a complaint.
    if (!(pendingValid_ && pendingRecord_.manual())) {
      for (auto& e : liquid_.trace) e = 0;
      liquid_.traceTurns = 0;
    }
  } else if (pendingValid_) {
    const uint32_t debt = saturate(static_cast<uint64_t>(liquid_.pendingDebt) + pendingCharge_);
    const uint32_t floor =
        saturate(static_cast<uint64_t>(liquid_.debtFloor) + permilleOf(pendingCharge_, kLiquidFloorPermille));
    liquid_.debtFloor = floor < debt ? floor : debt;
    if (liquid_.tilesActive || liquid_.tilesStaged) {
      for (uint8_t i = 0; i < kTileCount; ++i) liquid_.tile[i] = static_cast<int16_t>(tileNext(i));
      liquid_.tilesActive = true;
    }
    if (pendingFrameValid_) {
      // e = gamma*lambda*e + (a - p)*psi with a = 0: the policy chose fast.
      for (int i = 0; i < 4; ++i) {
        liquid_.trace[i] =
            addSat16(liquid_.trace[i] * kTraceDecayPermille / 1000, -liquid_.pressure * liquid_.psi[i] / 1000);
      }
      if (liquid_.traceTurns != 0xFF) ++liquid_.traceTurns;
    }
  }
  for (auto& s : liquid_.staged) s = 0;
  liquid_.tilesStaged = false;
  liquid_.stagedElapsedMs = 0;
}

bool AdaptiveRefreshPolicy::cfcDecide(const Inputs& in) {
  const auto& f = in.frame;
  uint8_t x[kCfcInputs] = {};
  uint64_t changedTiles = 0;
  if (pendingFrameValid_) {
    const uint64_t changed = f.historyValid ? f.changedPixels : estimateChangedPixels(f);
    const uint64_t share = changed * 128 / f.pixelCount;
    x[kCfcChanged] = static_cast<uint8_t>(share > 128 ? 128 : share);
    if (liquid_.tilesStaged) {
      for (uint8_t i = 0; i < kTileCount; ++i) {
        if (liquid_.staged[i] >= 2) changedTiles |= uint64_t{1} << i;  // >= 8 permille of the tile
      }
    } else if (changed != 0) {
      changedTiles = kAllTiles;
    }
  }
  const uint32_t elapsed = elapsedFor(in);
  if (elapsed != 0 && elapsed < kFastTurnMs) x[kCfcFastTurn] = static_cast<uint8_t>((kFastTurnMs - elapsed) * 128 / kFastTurnMs);
  x[kCfcImageArea] = in.hasImage ? 64 : 0;
  x[kCfcDither] = in.visibleGrayscale ? 32 : 0;
  if (lastImageTileMask_ != 0 && changedTiles != 0) {
    const uint32_t covered = static_cast<uint32_t>(__builtin_popcountll(lastImageTileMask_));
    const uint32_t hit = static_cast<uint32_t>(__builtin_popcountll(lastImageTileMask_ & changedTiles));
    x[kCfcOverlap] = static_cast<uint8_t>((static_cast<uint32_t>(cfcImageAreaPermille_) * 128 * hit + covered * 500) /
                                          (1000 * covered));
    if (hit != 0 && x[kCfcOverlap] == 0) x[kCfcOverlap] = 1;
  }
  cfcInverted_ = in.invertedOutput;
  x[kCfcInverted] = in.invertedOutput ? 128 : 0;
  // The exposure step already covered cfcExposedMs_ of this dwell.
  const uint32_t dt = elapsed > cfcExposedMs_ ? elapsed - cfcExposedMs_ : 0;
  cfcStep(cfcHiddenQ8_, x, dt, cfcPendingQ8_);
  cfcLogit_ = cfcReadoutOf(cfcReadoutW0_, cfcReadoutB0_, cfcPendingQ8_);
  return pendingFrameValid_ && x[kCfcOverlap] != 0 && cfcLogit_ >= 0;
}

// Runs before liquidCommit(): reads the pending decide() state.
void AdaptiveRefreshPolicy::cfcCommit(const Refresh displayed) {
  if (pendingValid_) {
    memcpy(cfcHiddenQ8_, cfcPendingQ8_, sizeof cfcHiddenQ8_);
    cfcExposedMs_ = 0;
  }
  if (displayed == Refresh::Clear) {
    // The clear rewrites every pixel: charge and polarization start over.
    const bool manual = pendingValid_ && pendingRecord_.manual();
    int16_t keep[kCfcUnits];
    memcpy(keep, cfcTraceQ8_, sizeof keep);
    const int16_t keepBias = cfcTraceBiasQ8_;
    const uint8_t keepTurns = cfcTraceTurns_;
    const int32_t logit = cfcLogit_;
    const bool inverted = cfcInverted_;
    resetCfcEpisode();
    cfcLogit_ = logit;
    cfcInverted_ = inverted;
    // A manual clear keeps the trace for recordManualRefresh(); any other
    // clear ends the episode.
    if (manual) {
      memcpy(cfcTraceQ8_, keep, sizeof keep);
      cfcTraceBiasQ8_ = keepBias;
      cfcTraceTurns_ = keepTurns;
    }
    return;
  }
  if (displayed == Refresh::Region) {
    // The picture region was cleared: its polarization is gone.
    for (uint8_t j = kCfcStickFirst; j < kCfcStickFirst + kCfcGroupUnits; ++j) cfcHiddenQ8_[j] = 0;
    lastImageTileMask_ = 0;
    cfcImageAreaPermille_ = 0;
  }
  if (pendingValid_ && pendingFrameValid_) {
    // e = gamma*lambda*e + dy0/dw: y0 is linear in w, so the gradient is h.
    for (uint8_t j = 0; j < kCfcUnits; ++j) {
      cfcTraceQ8_[j] = addSat16(cfcTraceQ8_[j] * kCfcTraceDecayPermille / 1000, cfcHiddenQ8_[j]);
    }
    cfcTraceBiasQ8_ = addSat16(cfcTraceBiasQ8_ * kCfcTraceDecayPermille / 1000, 256);
    if (cfcTraceTurns_ != 0xFF) ++cfcTraceTurns_;
  }
}

void AdaptiveRefreshPolicy::recordImageExposure(const uint32_t dwellMs, uint16_t areaPermille,
                                                uint16_t ditherPermille, const uint64_t coveredTileMask) {
  if (areaPermille == 0) return;
  if (areaPermille > 1000) areaPermille = 1000;
  if (ditherPermille > 1000) ditherPermille = 1000;
  uint8_t x[kCfcInputs] = {};
  x[kCfcImageArea] = q7FromPermille(areaPermille);
  x[kCfcDither] = q7FromPermille(ditherPermille);
  x[kCfcInverted] = cfcInverted_ ? 128 : 0;
  int16_t next[kCfcUnits];
  cfcStep(cfcHiddenQ8_, x, dwellMs, next);
  memcpy(cfcHiddenQ8_, next, sizeof next);
  const uint64_t tiles = coveredTileMask & kAllTiles;
  lastImageTileMask_ |= tiles ? tiles : kAllTiles;
  if (areaPermille > cfcImageAreaPermille_) cfcImageAreaPermille_ = areaPermille;
  cfcExposedMs_ = dwellMs > kMaxU32 - cfcExposedMs_ ? kMaxU32 : cfcExposedMs_ + dwellMs;
}

AdaptiveRefreshPolicy::DitherParams AdaptiveRefreshPolicy::computeImageDitherParams(
    const uint16_t edgeDensityPermille, const uint16_t midGrayPermille) const {
  uint8_t x[kCfcInputs] = {};
  x[kCfcImageArea] = 128;
  x[kCfcDither] = q7FromPermille(midGrayPermille);
  x[kCfcInverted] = cfcInverted_ ? 128 : 0;
  const uint32_t edge = static_cast<uint32_t>(edgeDensityPermille) * 128 / kCfcEdgeFullPermille;
  x[kCfcEdge] = static_cast<uint8_t>(edge > 128 ? 128 : edge);
  x[kCfcMidGray] = q7FromPermille(midGrayPermille);
  int16_t h[kCfcUnits];
  cfcStep(cfcHiddenQ8_, x, kCfcDitherProbeMs, h);
  DitherParams p;
  p.contrastGainQ8 = static_cast<uint16_t>(
      clampI(256 + cfcReadoutOf(kCfcWout[1], kCfcBout[1], h) / 64, kContrastGainMinQ8, kContrastGainMaxQ8));
  p.errorBleedQ8 = static_cast<uint16_t>(
      clampI(256 + cfcReadoutOf(kCfcWout[2], kCfcBout[2], h) / 64, kErrorBleedMinQ8, kErrorBleedMaxQ8));
  return p;
}

uint64_t AdaptiveRefreshPolicy::tileMaskFor(const Rect rect, const uint16_t screenWidth, const uint16_t screenHeight) {
  if (screenWidth == 0 || screenHeight == 0 || rect.w <= 0 || rect.h <= 0) return 0;
  const int32_t x0 = clampI(rect.x, 0, screenWidth - 1);
  const int32_t y0 = clampI(rect.y, 0, screenHeight - 1);
  const int32_t x1 = clampI(static_cast<int32_t>(rect.x) + rect.w - 1, 0, screenWidth - 1);
  const int32_t y1 = clampI(static_cast<int32_t>(rect.y) + rect.h - 1, 0, screenHeight - 1);
  if (x1 < x0 || y1 < y0) return 0;
  uint64_t mask = 0;
  for (int32_t row = y0 * kTileRows / screenHeight; row <= y1 * kTileRows / screenHeight; ++row) {
    for (int32_t col = x0 * kTileCols / screenWidth; col <= x1 * kTileCols / screenWidth; ++col) {
      mask |= uint64_t{1} << (row * kTileCols + col);
    }
  }
  return mask;
}

// e-prop on y0's readout. The complaint says the gate should have opened
// (target 1), so w += eta * (1 - p) * e with p the gate's hard-sigmoid
// probability at the last decide().
bool AdaptiveRefreshPolicy::cfcLearn() {
  if (cfcTraceTurns_ == 0) return false;
  const int32_t p = clampI(128 + cfcLogit_ / 64, 0, 256);  // Q8; 0 at y0 <= -2, 1 at y0 >= +2
  const int64_t loss = 256 - p;
  const auto step = [loss](const int32_t e) {
    const int64_t v = static_cast<int64_t>(e) * loss * kCfcLearnRate;
    return static_cast<int32_t>((v + (v >= 0 ? (1 << 19) : -(1 << 19))) / (1 << 20));
  };
  const CfcReadout& fitted = kCfcFitted;
  for (uint8_t j = 0; j < kCfcUnits; ++j) {
    const int32_t w = clampI(cfcReadoutW0_[j] + step(cfcTraceQ8_[j]), fitted.w[j] - kCfcReadoutDrift,
                             fitted.w[j] + kCfcReadoutDrift);
    cfcReadoutW0_[j] = static_cast<int8_t>(clampI(w, -127, 127));
    cfcTraceQ8_[j] = 0;
  }
  cfcReadoutB0_ = static_cast<int16_t>(clampI(cfcReadoutB0_ + step(cfcTraceBiasQ8_), fitted.b - kCfcReadoutDrift,
                                              fitted.b + kCfcReadoutDrift));
  cfcTraceBiasQ8_ = 0;
  cfcTraceTurns_ = 0;
  if (cfcUpdates_ != kMaxU32) ++cfcUpdates_;
  return true;
}

void AdaptiveRefreshPolicy::recordTileTransitions(const uint16_t tileChanged[kTileCount], const uint32_t elapsedMs) {
  for (uint8_t i = 0; i < kTileCount; ++i) {
    const uint16_t v = tileChanged[i] > 1000 ? 1000 : tileChanged[i];
    liquid_.staged[i] = static_cast<uint8_t>((v + 2) / 4);
  }
  liquid_.tilesStaged = true;
  liquid_.stagedElapsedMs = elapsedMs;
}

bool AdaptiveRefreshPolicy::recordManualRefresh() {
  const bool cfc = cfcLearn();
  if (liquid_.traceTurns == 0) return cfc;
  // m = 1: v -= eta * m * e. The trace holds -p*psi, so every feature active
  // before the complaint gains weight and the clear comes earlier.
  for (int i = 0; i < 4; ++i) {
    readout_.w[i] = addSat16(readout_.w[i], -liquid_.trace[i] * kLearnRatePermille / 1000);
    liquid_.trace[i] = 0;
  }
  liquid_.traceTurns = 0;
  clampReadout();
  if (liquid_.updates != kMaxU32) ++liquid_.updates;
  return true;
}

// Box bounds, then keep the plain debt threshold (1000 - b - v2*streak) / v0
// within [500, 1500] permille of the configured one for any streak.
void AdaptiveRefreshPolicy::clampReadout() {
  static constexpr int16_t kLo[4] = {667, 500, 0, -500};
  static constexpr int16_t kHi[4] = {2000, 2000, 500, 500};
  int16_t* const w = readout_.w;
  for (int i = 0; i < 4; ++i) w[i] = static_cast<int16_t>(clampI(w[i], kLo[i], kHi[i]));
  const int32_t lowest = 1000 - w[kReadoutStreak] - (w[kReadoutDebt] + 1) / 2;  // threshold >= 50%
  const int32_t highest = 1000 - w[kReadoutDebt] * 3 / 2;                        // threshold <= 150%
  w[kReadoutBias] = static_cast<int16_t>(clampI(w[kReadoutBias], highest, lowest));
}

uint32_t AdaptiveRefreshPolicy::learnedThresholdPermille(const bool inverted) const {
  const uint32_t threshold = inverted ? config_.nightClearThresholdPermille : config_.clearThresholdPermille;
  const uint64_t scaled = static_cast<uint64_t>(threshold) * static_cast<uint32_t>(1000 - readout_.w[kReadoutBias]);
  return static_cast<uint32_t>(scaled / static_cast<uint32_t>(readout_.w[kReadoutDebt]));
}

void AdaptiveRefreshPolicy::resetLearnedState() {
  readout_ = LiquidReadout{};
  liquid_.updates = 0;
  memcpy(cfcReadoutW0_, kCfcFitted.w, sizeof cfcReadoutW0_);
  cfcReadoutB0_ = kCfcFitted.b;
  cfcUpdates_ = 0;
}

size_t AdaptiveRefreshPolicy::saveLearnedState(uint8_t* buf, const size_t len) const {
  if (!buf || len < kLearnedStateBytes) return 0;
  LearnedFile file{};
  file.magic = kLearnedStateMagic;
  file.version = kLearnedStateVersion;
  file.bytes = kLearnedStateBytes;
  memcpy(file.w, readout_.w, sizeof file.w);
  file.updates = liquid_.updates;
  memcpy(file.cfcW0, cfcReadoutW0_, sizeof file.cfcW0);
  file.cfcB0 = cfcReadoutB0_;
  file.cfcUpdates = cfcUpdates_;
  file.check = fnv1a32(reinterpret_cast<const uint8_t*>(&file), offsetof(LearnedFile, check));
  memcpy(buf, &file, sizeof file);
  return kLearnedStateBytes;
}

bool AdaptiveRefreshPolicy::loadLearnedState(const uint8_t* buf, const size_t len) {
  if (!buf || len < kLearnedStateV1Bytes) return false;
  LearnedFileV1 head;
  memcpy(&head, buf, sizeof head);
  if (head.magic != kLearnedStateMagic) return false;
  if (head.version == 1 && head.bytes == kLearnedStateV1Bytes && len == kLearnedStateV1Bytes) {
    // An rc9/rc10-v1 file: the v1 readout only; the CfC readout stays as is.
    if (head.check != fnv1a32(buf, offsetof(LearnedFileV1, check))) return false;
    memcpy(readout_.w, head.w, sizeof head.w);
    liquid_.updates = head.updates;
    clampReadout();
    return true;
  }
  LearnedFile file;
  if (len < kLearnedStateBytes) return false;
  memcpy(&file, buf, sizeof file);
  if (file.version != kLearnedStateVersion || file.bytes != kLearnedStateBytes ||
      file.check != fnv1a32(buf, offsetof(LearnedFile, check))) {
    return false;
  }
  memcpy(readout_.w, file.w, sizeof file.w);
  liquid_.updates = file.updates;
  clampReadout();
  for (uint8_t j = 0; j < kCfcUnits; ++j) {
    cfcReadoutW0_[j] = static_cast<int8_t>(clampI(clampI(file.cfcW0[j], kCfcFitted.w[j] - kCfcReadoutDrift,
                                                         kCfcFitted.w[j] + kCfcReadoutDrift),
                                                  -127, 127));
  }
  cfcReadoutB0_ = static_cast<int16_t>(
      clampI(file.cfcB0, kCfcFitted.b - kCfcReadoutDrift, kCfcFitted.b + kCfcReadoutDrift));
  cfcUpdates_ = file.cfcUpdates;
  return true;
}

void AdaptiveRefreshPolicy::imagePageShown(const Rect pictures) {
  if (config_.clearAfterImage) {
    if (pictures.empty()) {
      imageRegionUnknown_ = true;
    } else if (imageRegion_.empty()) {
      imageRegion_ = pictures;
    } else {
      const int32_t x0 = imageRegion_.x < pictures.x ? imageRegion_.x : pictures.x;
      const int32_t y0 = imageRegion_.y < pictures.y ? imageRegion_.y : pictures.y;
      const int32_t x1 = imageRegion_.x + imageRegion_.w > pictures.x + pictures.w ? imageRegion_.x + imageRegion_.w
                                                                                   : pictures.x + pictures.w;
      const int32_t y1 = imageRegion_.y + imageRegion_.h > pictures.y + pictures.h ? imageRegion_.y + imageRegion_.h
                                                                                   : pictures.y + pictures.h;
      imageRegion_ = {static_cast<int16_t>(x0), static_cast<int16_t>(y0), static_cast<int16_t>(x1 - x0),
                      static_cast<int16_t>(y1 - y0)};
    }
    requestClear(ClearSource::Image);
    return;
  }
  debt_ = saturate(static_cast<uint64_t>(debt_) + clearedCharge_);
  liquid_.debtFloor =
      saturate(static_cast<uint64_t>(liquid_.debtFloor) + permilleOf(clearedCharge_, kLiquidFloorPermille));
  clearedCharge_ = 0;
}
