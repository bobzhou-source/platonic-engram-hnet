#pragma once

#include <FrameChangeMetrics.h>
#include <stddef.h>
#include <stdint.h>

// Content-aware "when does the next page need a clearing refresh" policy.
//
// Replaces a fixed page-count interval. Every displayed frame is charged an
// estimate of how many pixels the panel was asked to change, weighted by how
// hard that content is on the panel (images, visible grayscale). The charge
// accumulates as "ghost debt" that only a clearing refresh (HALF/FULL) erases.
// A page that changes little therefore lets many more fast refreshes through
// than a dense page or an image, but a conservative safety ceiling bounds the
// worst case. The compiled constants are engineering estimates, NOT a
// calibration against physical ghosting; a Config loaded from the SD card
// (see RefreshPolicyConfig.h) can replace them with calibrated values.
//
// Protocol (single owner, one frame at a time, no allocation, no I/O):
//   1. decide(inputs)   once per frame, before choosing the refresh mode. Pure
//                       with respect to debt: it only remembers this frame's
//                       charge so commit() can book it.
//   2. run the physical refresh with the returned mode (or whatever promotion
//      actually applies - manual/popup/inversion promotions are real clears).
//   3. commit(actual)   once, only if that refresh really reached the panel.
//                       Pass Clear whenever the hardware ran HALF or FULL,
//                       Region when a fast frame plus its windowed clear ran.
//      Not committing (display failed/aborted) leaves all state untouched: the
//      panel still shows the previous frame.
// Frames whose refresh mode is dictated elsewhere (overlays, menus) that should
// still count are accounted the same way: decide(), ignore the answer, commit().
//
// LiquidRefresh (optional, off until fed): a closed-form continuous-time
// (CfC) relaxation of the debt between turns plus a backprop-free
// eligibility-trace readout that learns from manual refreshes.
//   - Inputs::elapsedMs (or recordTileTransitions()) gives the page dwell.
//     Before the new charge is added the relaxing share of the debt decays by
//     rho = exp(-dt * softplus(f(polarity, class))); kLiquidFloorPermille of
//     every charge is irreversible until the next clear. Fast turns
//     (dt < kFastTurnMs) are charged up to kFastTurnBoostPermille extra.
//   - recordTileTransitions() feeds an 8x6 tile plane; a tile ghosting well
//     above the screen mean clears earlier than the same change spread out.
//   - The debt rule becomes the readout z = v.psi + b >= 1000 over
//     psi = [debt_ratio, peak_tile_excess, fast_streak, 1] (all permille).
//     recordManualRefresh() moves v, b along the eligibility trace
//     e = gamma*lambda*e + (a - p)*psi, bounded so the plain debt threshold
//     stays within [50%, 150%] of the configured one. The safety cap and all
//     other clear reasons are unchanged.
// With elapsedMs == 0, no tiles and the default readout, every decision,
// debt and clear row is the content-v2 rule exactly.
//
// LiquidRefresh CfC v2 (Hasani et al., arXiv:2106.13898): a 16-unit
// closed-form continuous-time cell over 8 screen/image inputs,
//   h_t = exp(-f(x_t) dt) * h_{t-1} + (1 - exp(-f(x_t) dt)) * tanh(W_g x_t + b_g),
//   f(x) = softplus(W_f x + b_f) per second,
// with three linear readouts over h: y0 the refresh gate, y1 the dither
// contrast gain and y2 the error-diffusion bleed. Units 0-5 model reversible
// text charge (rapid turns charge them, reading dwell relaxes them), units
// 6-11 image sticking (an image's viewing dwell polarizes them, text relaxes
// them slowly), units 12-15 are free. The int8 tables are fitted offline
// (offline fitter fit_liquid_cfc_v2.py) and exported into the .cpp.
//   - recordImageExposure() steps the cell over an image's viewing dwell and
//     remembers the tiles it covered. decide() steps it over the text page's
//     own dwell (exposed time is not counted twice) and answers
//     Reason::ImageSticking when y0 >= 0 AND the page changes tiles the image
//     covered. Without an exposure that reason never fires, so every
//     content-v2 / LiquidRefresh v1 decision above is unchanged.
//   - computeImageDitherParams() probes the cell with an image's edge and
//     mid-gray density (no state change) and returns y1, y2 for the decoders.
//   - y0's readout (16 weights + bias) learns by e-prop: a 16-dim eligibility
//     trace e = gamma*lambda*e + h over the fast turns since the last clear;
//     recordManualRefresh() moves w += eta * (1 - p) * e, bounded around the
//     fitted readout. It is persisted next to the v1 readout (LQRF v2).
class AdaptiveRefreshPolicy {
 public:
  // Clear maps to the app's HALF refresh (FULL stays a manual/GfxRenderer promotion).
  // Region: the frame is drawn fast, then a clearing waveform runs over
  // Decision::region only (Config::imageClearRegion). Returned only to callers
  // that set Inputs::regionClearAvailable.
  enum class Refresh : uint8_t { Fast, Clear, Region };

  enum class Reason : uint8_t {
    None,            // Fast: debt is within budget
    Debt,            // accumulated weighted change reached the budget
    Initial,         // nothing known about what the panel shows yet
    Requested,       // requestClear(): popup, post-image cleanup, buffer rebuild, ...
    External,        // a renderer-level clear promotion is already queued
    InvalidFrame,    // framebuffer unreadable or zero-sized panel metrics
    PolarityChange,  // output inversion flipped: the next refresh rewrites everything
    SafetyCap,       // kSafetyCap fast refreshes in a row; bound only, not the normal rule
    ImageSticking    // CfC v2: the page overwrites a polarized image (y0 >= 0)
  };

  // Who asked for a Requested/External clear. Bookkeeping for the usage log
  // only; it never changes a decision. Manual outranks every other source.
  enum class ClearSource : uint8_t { None, Other, Image, Popup, External, Manual };

  // Logical screen rectangle (the renderer's coordinates). No member
  // initializers (the class uses Rect{} inside its own definition); Rect{}
  // is the empty rect.
  struct Rect {
    int16_t x;
    int16_t y;
    int16_t w;
    int16_t h;
    bool empty() const { return w <= 0 || h <= 0; }
    bool operator==(const Rect&) const = default;
  };

  // Why a picture clear in region mode covered the whole screen. Persisted in
  // benchmark rows: append only.
  enum class RegionFallback : uint8_t {
    None,         // not region mode, or the region clear ran
    Unknown,      // a picture page did not say where its pictures were
    OffScreen,    // the pictures' area is empty once clamped to the screen
    TooLarge,     // the area exceeds kRegionMaxPermille of the screen
    Unavailable,  // the panel cannot run a windowed clear right now
    Overruled     // a stronger reason (debt, safety cap, polarity, ...) needs the whole screen
  };

  struct Inputs {
    freeink::FrameChangeMetrics frame;
    // Real page metadata (not a setting): the page contains a decoded image.
    bool hasImage = false;
    // A grayscale pass is actually planned for this frame AND the panel supports
    // it right now. Never set it just because AA is enabled or the panel is in
    // an inverted mode that renders BW only.
    bool visibleGrayscale = false;
    // A clearing refresh is already queued by the renderer (manual refresh,
    // popup close promotion); the frame is going to be a clear regardless.
    bool externalClearPending = false;
    // The panel output is inverted (night mode): the night threshold applies.
    bool invertedOutput = false;
    // The queued renderer promotion came from a user's refresh command (the
    // control center's refresh tile). Only labels the clear; see ClearRecord.
    bool externalManual = false;
    // The renderer can run a windowed clear after this frame
    // (GfxRenderer::prepareRegionClear() succeeded). Only read with
    // Config::imageClearRegion; leave false everywhere else.
    bool regionClearAvailable = false;
    // Logical screen size, to clamp the pictures' area.
    uint16_t screenWidth = 0;
    uint16_t screenHeight = 0;
    // Milliseconds the previous frame stayed on the panel (page dwell). 0:
    // unknown; the liquid decay and fast-turn boost are then off unless
    // recordTileTransitions() passed a dwell for this frame.
    uint32_t elapsedMs = 0;
  };

  struct Decision {
    Refresh refresh;
    Reason reason;
    Rect region{};  // Refresh::Region only: the logical rect to clear
  };

  // One committed clear, for the benchmark recorder's refresh rows. Counts
  // cover the fast refreshes since the previous committed clear.
  struct ClearRecord {
    Reason reason = Reason::None;  // None: the caller cleared although decide() said fast
    ClearSource source = ClearSource::None;
    uint16_t fastTurns = 0;   // fast refreshes since the previous clear (saturating)
    uint16_t imagePages = 0;  // of those, frames with a page image
    uint16_t grayPages = 0;   // of those, frames with a visible grayscale pass
    uint32_t debtPermille = 0;       // debt plus this frame's charge, thousandths of a screen
    uint32_t thresholdPermille = 0;  // clear threshold in force for this frame's polarity
    bool inverted = false;
    bool image = false;      // the clearing frame itself has a page image
    bool grayscale = false;  // the clearing frame itself has a visible grayscale pass
    // A region clear: only `region` was cleared and the counts above carry on
    // to the next full clear (it does not end the debt window).
    bool regionOnly = false;
    RegionFallback regionFallback = RegionFallback::None;
    Rect region{};
    bool manual() const { return source == ClearSource::Manual; }
  };
  // Called from commit(Clear) and commit(Region) only; never on a fast
  // refresh. No I/O expected.
  using ClearObserver = void (*)(void* context, const ClearRecord& record);

  // --- Identity: bump BOTH whenever a threshold, weight or rule below changes, so
  // benchmark/settings metadata can never describe a different policy than the one
  // that ran. A loaded Config is reported separately (RefreshPolicyConfig).
  static constexpr uint8_t kVersion = 2;
  static constexpr char kName[] = "adaptive-content-v2";

  // --- Compiled defaults (public so docs and tests can cite them) ----------
  // Clear when accumulated weighted change reaches this many thousandths of a
  // full screen of changed pixels (1500 = 1.5 screens).
  static constexpr uint32_t kClearThresholdPermille = 1500;
  // Hard ceiling on consecutive fast refreshes. Far above what normal reading
  // reaches (sparse pages need ~tens of turns); it only stops pathological
  // streams of tiny changes from never clearing.
  static constexpr uint16_t kSafetyCap = 60;
  // Charge multiplier in thousandths (1000 = 1.0x): a visible grayscale pass
  // adds 0.25x (AA text alone is not heavy), a page image adds 1.0x, and a
  // grayscale image adds a further 0.5x on top of both.
  static constexpr uint32_t kWeightUnitPermille = 1000;
  static constexpr uint32_t kWeightGrayscalePermille = 250;
  static constexpr uint32_t kWeightImagePermille = 1000;
  static constexpr uint32_t kWeightGrayscaleImageExtraPermille = 500;
  // Flat charge per fast frame, in thousandths of a screen, independent of
  // how much changed. 0: debt follows changed pixels only.
  static constexpr uint32_t kFrameChargePermille = 0;
  // The page after an inline picture is a clear. With false the picture page
  // is charged like any other frame (the image weights above) and the clear
  // waits for the debt threshold.
  static constexpr bool kClearAfterImage = true;
  // Region mode (only with clearAfterImage): the clear after a picture page
  // runs over the union of the pictures' rects instead of the whole screen,
  // unless that union exceeds this share of the screen. Text debt, the safety
  // cap and every other clear reason still clear the whole screen.
  static constexpr bool kImageClearRegion = false;
  static constexpr uint32_t kRegionMaxPermille = 600;

  // How EPUB picture pages are drawn on panels with absolute grayscale planes
  // (UC8279). Quality: absolute four-grey planes on the quality waveform,
  // which cycles the whole page. Fast: the B/W page plus overlay grey masks;
  // the driver still picks the quality waveform when greys cover a large part
  // of the panel. Panels without absolute planes always use overlay masks.
  enum class ImageWaveform : uint8_t { Quality, Fast };
  static constexpr ImageWaveform kImageWaveform = ImageWaveform::Quality;

  // --- LiquidRefresh constants --------------------------------------------
  static constexpr uint8_t kTileCols = 8;
  static constexpr uint8_t kTileRows = 6;
  static constexpr uint8_t kTileCount = kTileCols * kTileRows;
  // Share of each charge that never relaxes before a clear.
  static constexpr uint32_t kLiquidFloorPermille = 500;
  // f(polarity, class) in thousandths; the relaxation rate is softplus(f) per
  // second (evaluated at compile time): text ~0.049/s (tau ~21 s), grayscale
  // ~0.030/s, image ~0.018/s. Night mode (inverted output) adds
  // kLiquidLogitInvertedMilli: slower. Classes index kLiquidLogitMilli.
  enum LiquidClass : uint8_t { kLiquidText, kLiquidGray, kLiquidImage };
  static constexpr int32_t kLiquidLogitMilli[3] = {-3000, -3500, -4000};
  static constexpr int32_t kLiquidLogitInvertedMilli = -700;
  // A dwell below kFastTurnMs is a fast turn: its charge grows linearly up to
  // +kFastTurnBoostPermille as the dwell approaches 0 and it extends the
  // fast streak (full at kFastStreakFull turns).
  static constexpr uint32_t kFastTurnMs = 2000;
  static constexpr uint32_t kFastTurnBoostPermille = 250;
  static constexpr uint8_t kFastStreakFull = 8;
  // A tile ghosting kTilePeakScale x the clear threshold above the screen
  // mean contributes a full 1000 to the readout.
  static constexpr uint32_t kTilePeakScale = 2;
  // Eligibility trace decay (gamma * lambda) and learning rate, thousandths.
  static constexpr int32_t kTraceDecayPermille = 750;
  static constexpr int32_t kLearnRatePermille = 50;
  // Learned-state file: 64 bytes, "LQRF" on disk (little-endian magic). v2
  // keeps the v1 layout in its first 28 bytes and adds the CfC readout; a
  // 32-byte v1 file still loads (the CfC readout then stays the fitted one).
  static constexpr size_t kLearnedStateBytes = 64;
  static constexpr size_t kLearnedStateV1Bytes = 32;
  static constexpr uint32_t kLearnedStateMagic = 0x4652514Cu;
  static constexpr uint16_t kLearnedStateVersion = 2;
  static constexpr char kLearnedStatePath[] = "/.crosspoint/liquid-refresh.bin";

  // --- LiquidRefresh CfC v2 constants -------------------------------------
  static constexpr uint8_t kCfcInputs = 8;
  static constexpr uint8_t kCfcUnits = 16;
  static constexpr uint8_t kCfcOutputs = 3;
  // Inputs, Q7 (128 = 1.0): share of the screen the frame changes, fast-turn
  // intensity (1 at dt -> 0, 0 from kFastTurnMs), image area, dither (mid-gray)
  // density, image-to-text overlap (area x share of its tiles the page
  // changes), night output, edge density (1.0 at kCfcEdgeFullPermille) and
  // mid-gray density.
  enum CfcInput : uint8_t {
    kCfcChanged,
    kCfcFastTurn,
    kCfcImageArea,
    kCfcDither,
    kCfcOverlap,
    kCfcInverted,
    kCfcEdge,
    kCfcMidGray
  };
  static constexpr uint8_t kCfcChargeFirst = 0;  // units 0-5: reversible text charge
  static constexpr uint8_t kCfcStickFirst = 6;   // units 6-11: image sticking
  static constexpr uint8_t kCfcGroupUnits = 6;
  static constexpr uint16_t kCfcEdgeFullPermille = 250;
  // Dwell the dither probe integrates an image over, and the longest step.
  static constexpr uint32_t kCfcDitherProbeMs = 6000;
  static constexpr uint32_t kCfcMaxStepMs = 600000;
  // y1, y2 -> Q8 parameters: 256 + Y / 64, clamped.
  static constexpr uint16_t kContrastGainMinQ8 = 240;
  static constexpr uint16_t kContrastGainMaxQ8 = 312;
  static constexpr uint16_t kErrorBleedMinQ8 = 160;
  static constexpr uint16_t kErrorBleedMaxQ8 = 256;
  // e-prop: trace decay (gamma * lambda, thousandths), learning rate, and how
  // far the learned y0 readout may move from the fitted one (1/16 units).
  static constexpr int32_t kCfcTraceDecayPermille = 750;
  static constexpr int32_t kCfcLearnRate = 16;
  static constexpr int16_t kCfcReadoutDrift = 48;

  struct DitherParams {
    uint16_t contrastGainQ8 = 256;  // 256 = 1.0x around mid-gray 128
    uint16_t errorBleedQ8 = 256;    // 256 = full error diffusion
    bool operator==(const DitherParams&) const = default;
  };
  // The learnable y0 readout: weights and bias in 1/16 units.
  struct CfcReadout {
    int8_t w[kCfcUnits];
    int16_t b;
    bool operator==(const CfcReadout&) const = default;
  };
  static const CfcReadout& cfcFittedReadout();

  // Readout weights, thousandths. The default makes z the plain debt ratio.
  enum LiquidWeight : uint8_t { kReadoutDebt, kReadoutPeak, kReadoutStreak, kReadoutBias };
  struct LiquidReadout {
    // v0 [667, 2000], v1 [500, 2000], v2 [0, 500], b [-500, 500] and then the
    // plain debt threshold bounds.
    int16_t w[4] = {1000, 1000, 0, 0};
    bool operator==(const LiquidReadout&) const = default;
  };

  // Every tunable of the rule. The defaults are the compiled constants above;
  // a policy built with Config{} behaves exactly like the compiled policy.
  struct Config {
    uint32_t clearThresholdPermille = kClearThresholdPermille;
    // Threshold used while invertedOutput is set (night mode).
    uint32_t nightClearThresholdPermille = kClearThresholdPermille;
    uint16_t safetyCap = kSafetyCap;
    uint32_t weightGrayscalePermille = kWeightGrayscalePermille;
    uint32_t weightImagePermille = kWeightImagePermille;
    uint32_t weightGrayscaleImageExtraPermille = kWeightGrayscaleImageExtraPermille;
    uint32_t frameChargePermille = kFrameChargePermille;
    bool clearAfterImage = kClearAfterImage;
    ImageWaveform imageWaveform = kImageWaveform;
    bool imageClearRegion = kImageClearRegion;

    bool operator==(const Config&) const = default;
  };

  // initialClearRequired=false is the quick-resume case where the panel is
  // already known to show the restored page (allowFastInitialRefresh).
  explicit AdaptiveRefreshPolicy(bool initialClearRequired = true) : AdaptiveRefreshPolicy(initialClearRequired, Config{}) {}
  explicit AdaptiveRefreshPolicy(bool initialClearRequired, const Config& config) : config_(config) {
    resetLearnedState();
    reset(initialClearRequired);
  }

  void reset(bool initialClearRequired = true);

  const Config& config() const { return config_; }

  // Sticky until a Clear is committed. Not charged as debt.
  void requestClear(ClearSource source = ClearSource::Other) {
    clearRequested_ = true;
    if (clearSource_ != ClearSource::Manual) clearSource_ = source;
  }
  void setClearObserver(ClearObserver observer, void* context) {
    observer_ = observer;
    observerContext_ = context;
  }

  // Call right after committing a page with an inline picture. With
  // clearAfterImage the next page is a requested clear. Without it the
  // picture's charge stays in the debt; when the picture page was itself the
  // clear, its greys were laid down after the clearing base, so its charge
  // opens the new budget instead of vanishing with the cleared debt.
  // `pictures` is the logical rect the page's pictures cover; empty means
  // unknown (region mode then clears the whole screen). Rects of picture
  // pages shown before the clear runs accumulate.
  void imagePageShown(Rect pictures = {});

  // The next decide() may answer Region: the pending clear is the picture
  // clear and region mode is on. Ask the renderer to prepare a windowed
  // clear (and set Inputs::regionClearAvailable) only then.
  bool regionClearWanted() const {
    return config_.imageClearRegion && config_.clearAfterImage && !initialClear_ && clearRequested_ &&
           clearSource_ == ClearSource::Image;
  }

  // The previous frame's ink is unknown (frames the policy never saw, buffer
  // lend/return). The fallback estimate then assumes the previous frame had the
  // current frame's density. Does not force a clear.
  void invalidateHistory() { prevInkKnown_ = false; }

  Decision decide(const Inputs& in);
  void commit(Refresh displayed);

  // Per-tile share of pixels (permille of that tile, row-major 8x6) the frame
  // about to be decided changes, plus its page dwell. Call before decide();
  // consumed by the next commit(). Inputs::elapsedMs, when set, wins.
  void recordTileTransitions(const uint16_t tileChanged[kTileCount], uint32_t elapsedMs);
  // The user asked for a refresh (m = 1): the clear came too late. Moves the
  // v1 readout along its eligibility trace and the CfC y0 readout along its
  // 16-dim e-prop trace of the fast turns since the last clear. Valid before
  // or after the manual clear is committed. False when no fast turn was traced.
  bool recordManualRefresh();

  // CfC v2: an image stayed on the panel for dwellMs (picture viewer closed,
  // or the reader turns away from a page with pictures). areaPermille: its
  // share of the screen; ditherPermille: its dithered/mid-gray share;
  // coveredTileMask: bit (row * kTileCols + col) per tile it covered, 0 =
  // unknown (every tile). Call before the next decide(); the step is booked
  // at once and decide() does not count dwellMs again.
  void recordImageExposure(uint32_t dwellMs, uint16_t areaPermille, uint16_t ditherPermille,
                           uint64_t coveredTileMask = 0);
  // y1, y2 for an image about to be dithered, from the current cell state and
  // the image's edge density (strong horizontal transitions) and mid-gray
  // density, both in permille of its pixels. Pure.
  DitherParams computeImageDitherParams(uint16_t edgeDensityPermille, uint16_t midGrayPermille) const;
  // Tiles (8x6 over a screenWidth x screenHeight logical screen) `rect` touches.
  static uint64_t tileMaskFor(Rect rect, uint16_t screenWidth, uint16_t screenHeight);
  int16_t cfcHiddenQ8(uint8_t unit) const { return unit < kCfcUnits ? cfcHiddenQ8_[unit] : 0; }
  // Mean of the charge (0-5) and image-sticking (6-11) units, Q8.
  int16_t cfcChargeQ8() const { return cfcGroupMean(kCfcChargeFirst); }
  int16_t cfcStickQ8() const { return cfcGroupMean(kCfcStickFirst); }
  // y0 of the last decide(), 1/4096 units: the gate opens at >= 0.
  int32_t cfcRefreshLogit() const { return cfcLogit_; }
  int16_t cfcTraceQ8(uint8_t unit) const { return unit < kCfcUnits ? cfcTraceQ8_[unit] : 0; }
  CfcReadout cfcReadout() const;
  uint32_t cfcUpdates() const { return cfcUpdates_; }
  uint64_t imageTileMask() const { return lastImageTileMask_; }
  // Learned readout persistence (kLearnedStatePath). No I/O: the caller owns
  // the file. save returns kLearnedStateBytes, or 0 when len is too small;
  // load rejects a short, foreign or corrupt buffer and keeps the state.
  size_t saveLearnedState(uint8_t* buf, size_t len) const;
  bool loadLearnedState(const uint8_t* buf, size_t len);
  void resetLearnedState();
  const LiquidReadout& liquidReadout() const { return readout_; }
  uint32_t liquidUpdates() const { return liquid_.updates; }
  // Plain debt threshold the readout implies (no tile peak, no fast streak).
  uint32_t learnedThresholdPermille(bool inverted = false) const;
  int16_t tileGhostPermille(uint8_t tile) const { return tile < kTileCount ? liquid_.tile[tile] : 0; }
  // rho in thousandths for a dwell over a resident frame of class `cls`:
  // exp(-elapsedMs/1000 * softplus(f(inverted, cls))), integer rational.
  static uint16_t liquidRhoPermille(uint32_t elapsedMs, uint8_t cls, bool inverted);

  bool clearPending() const { return initialClear_ || clearRequested_; }
  uint32_t debtPixels() const { return debt_; }
  uint16_t fastSinceClear() const { return fastSinceClear_; }
  // Charge decide() computed for the frame awaiting commit().
  uint32_t pendingCharge() const { return pendingCharge_; }

  // Changed-pixel estimate used when the display has no exact last-displayed
  // baseline (single framebuffer without a valid async shadow, borrowed or
  // released secondary buffer, startup, inverted output). With P = pixelCount,
  // ink = logical-ink pixels, fg(x) = min(x, P - x) (the sparse side, so a
  // dark-background theme does not look "all ink"), and maj(x) = (2x > P):
  //   same background polarity:  min(P, fg(cur) + fg(prev))
  //   flipped polarity:          P - min(P, fg(cur) + fg(prev))
  // i.e. an upper bound that treats consecutive pages as non-overlapping, which
  // is accurate for page turns and conservative for everything else. prev is
  // the ink recorded at the last commit, or cur when unknown.
  uint32_t estimateChangedPixels(const freeink::FrameChangeMetrics& frame) const;

 private:
  uint32_t chargeFor(const Inputs& in) const;
  uint32_t weightFor(const Inputs& in) const;
  uint32_t elapsedFor(const Inputs& in) const { return in.elapsedMs ? in.elapsedMs : liquid_.stagedElapsedMs; }
  bool liquidDebtDue(const Inputs& in);
  int32_t tileNext(uint8_t tile) const;
  void liquidCommit(Refresh displayed);
  void clampReadout();
  ClearRecord labelFor(const Inputs& in, Reason reason) const;
  RegionFallback regionFor(const Inputs& in, bool debtDue, Rect& region) const;
  bool cfcDecide(const Inputs& in);
  void cfcCommit(Refresh displayed);
  bool cfcLearn();
  void resetCfcEpisode();
  int16_t cfcGroupMean(uint8_t first) const;

  Config config_;
  uint32_t debt_ = 0;            // weighted changed pixels since the last clear (saturating)
  uint32_t pendingCharge_ = 0;   // this frame's charge, set by decide()
  uint32_t pendingInk_ = 0;      // this frame's logical-ink pixels
  uint32_t prevInk_ = 0;         // ink of the last committed frame
  uint32_t clearedCharge_ = 0;   // charge of the last committed frame if it was a clear, else 0
  uint16_t fastSinceClear_ = 0;  // saturating
  bool initialClear_ = true;
  bool clearRequested_ = false;
  bool pendingValid_ = false;
  bool pendingFrameValid_ = false;
  bool prevInkKnown_ = false;

  // Usage-log bookkeeping (never read by decide()).
  ClearObserver observer_ = nullptr;
  void* observerContext_ = nullptr;
  ClearRecord pendingRecord_;  // this frame's labels, set by decide()
  ClearSource clearSource_ = ClearSource::None;
  uint16_t imagePages_ = 0;
  uint16_t grayPages_ = 0;

  // Region mode: union of the rects of picture pages since the last clear.
  Rect imageRegion_{};
  bool imageRegionUnknown_ = false;

  // LiquidRefresh state, 232 bytes at most together with readout_. Episodic
  // fields are set by reset(); the readout survives it.
  struct Liquid {
    int16_t tile[kTileCount];        // tile ghost, thousandths of the tile
    uint8_t staged[kTileCount];      // recordTileTransitions(), units of 4 permille
    int16_t trace[4];                // eligibility trace e
    int16_t psi[4];                  // this frame's features, set by decide()
    uint32_t pendingDebt;            // decayed debt before this frame's charge
    uint32_t debtFloor;              // irreversible share of debt_
    uint32_t stagedElapsedMs;
    uint32_t updates;                // manual-refresh updates applied (saturating)
    uint16_t tileGain;               // this frame's tile charge multiplier, thousandths
    uint16_t retention;              // this frame's tile retention, thousandths
    int16_t pressure;                // p = clamp(z, 0, 1000)
    uint8_t fastStreak;
    uint8_t pendingStreak;
    uint8_t residentClass;           // LiquidClass of the frame on the panel
    uint8_t pendingClass;
    uint8_t traceTurns;              // fast turns in the trace (saturating)
    bool tilesActive;
    bool tilesStaged;
  };
  Liquid liquid_{};
  LiquidReadout readout_;

  // LiquidRefresh CfC v2 state (Q8 = 256 per 1.0). Episodic fields are reset
  // by reset() and by every committed clear; the readout survives both.
  int16_t cfcHiddenQ8_[kCfcUnits] = {};   // h, the cell state on the panel
  int16_t cfcPendingQ8_[kCfcUnits] = {};  // h after this frame, set by decide()
  int16_t cfcTraceQ8_[kCfcUnits] = {};    // e-prop eligibility trace of y0's weights
  int16_t cfcTraceBiasQ8_ = 0;            // ... and of its bias
  int8_t cfcReadoutW0_[kCfcUnits] = {};   // y0 readout, learnable (set by the ctor)
  int16_t cfcReadoutB0_ = 0;
  uint64_t lastImageTileMask_ = 0;        // tiles of images recorded since the last clear
  uint32_t cfcExposedMs_ = 0;             // exposure dwell decide() must not count again
  uint32_t cfcUpdates_ = 0;               // e-prop updates applied (saturating)
  int32_t cfcLogit_ = 0;                  // y0 of the last decide()
  uint16_t cfcImageAreaPermille_ = 0;     // largest image area recorded since the last clear
  uint8_t cfcTraceTurns_ = 0;             // fast turns in the trace (saturating)
  bool cfcInverted_ = false;              // night output at the last decide()
};
