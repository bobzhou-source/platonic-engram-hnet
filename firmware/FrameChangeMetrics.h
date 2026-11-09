#pragma once

#include <stdint.h>

namespace freeink {

// What FreeInkDisplay::measureFrameChange() knows about the frame currently in the
// framebuffer relative to the last frame handed to the panel. Counts are logical
// pixels: "ink" is a 0 bit (black) in the 1=white/0=black framebuffer, independent
// of output inversion (the framebuffer stays logical; inversion is applied only on
// the wire). Row padding bits beyond the panel width are never counted.
struct FrameChangeMetrics {
  uint32_t pixelCount = 0;         // visible pixels (width * height)
  uint32_t inkPixels = 0;          // ink pixels in the current framebuffer
  uint32_t previousInkPixels = 0;  // ink pixels in the baseline; 0 unless historyValid
  uint32_t changedPixels = 0;      // pixels differing from the baseline; 0 unless historyValid

  // The framebuffer was readable. False while the buffer is lent out, released,
  // or before begin(); every other field is then zero.
  bool frameValid = false;

  // An exact copy of the last displayed frame exists, so changedPixels and
  // previousInkPixels are exact. False when the build keeps no such copy (single
  // framebuffer without a valid async shadow), or when the baseline was
  // invalidated: blocking/gray/partial-window displays, inversion changes,
  // buffer release/realloc/lend, a borrowed secondary buffer, an uncommitted
  // async refresh. Callers must not read the two change counts as zero then.
  bool historyValid = false;

  // setInverted() changed the output polarity since the last display, so the next
  // refresh rewrites the whole panel regardless of how much content changed.
  bool polarityChangePending = false;
};

}  // namespace freeink
