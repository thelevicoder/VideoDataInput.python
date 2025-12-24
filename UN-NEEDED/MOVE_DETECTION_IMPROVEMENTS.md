# MOVE DETECTION IMPROVEMENTS

## Problem Summary

The original move detection was too sensitive and detected false moves due to:

1. **Too Low Thresholds**
   - `MIN_SEG_FRAMES = 3` - Too short, counted jitter as moves
   - `MIN_HOLD_SWITCH_DIST = 8.0` pixels - Tiny movements counted as moves
   - No velocity requirements

2. **No Motion Analysis**
   - Didn't check if limb actually moved significantly
   - No distinction between real moves and micro-adjustments
   - Counted oscillations/jitter as multiple moves

3. **Oversimplified Logic**
   - Just checked if hold ID changed
   - No stability requirements
   - No filtering of back-and-forth movements

## Key Improvements

### 1. Stricter Stability Requirements

**OLD:**
- `MIN_SEG_FRAMES = 3` (0.1 seconds @ 30fps)

**NEW:**
- `MIN_STABLE_FRAMES = 8` (0.27 seconds @ 30fps)
- Requires sustained low velocity while on hold
- Limb must truly settle before counting as stable

### 2. Significant Movement Only

**OLD:**
- `MIN_HOLD_SWITCH_DIST = 8.0` pixels
- No velocity checks

**NEW:**
- `MIN_HOLD_SWITCH_DIST = 60.0` pixels (base)
- `MIN_MOVE_VELOCITY = 15.0` pixels/frame
- Resolution-adaptive thresholds
- Requires both distance AND velocity

### 3. Velocity-Based Detection

**NEW Features:**
- Calculates frame-by-frame velocity for each limb
- Tracks average and max velocity during transitions
- Only counts moves with sufficient velocity
- Filters out slow drifting/adjustments

### 4. Oscillation Filtering

**NEW Features:**
- Detects back-and-forth patterns (A→B→A)
- Filters out jitter and micro-adjustments
- Only counts if limb stays on new hold long enough

### 5. Resolution Adaptation

**NEW Features:**
- Automatically adjusts thresholds based on video resolution
- Higher resolution = proportionally larger thresholds
- Works well from 720p to 4K

## Usage

### Option 1: Replace Original File

```bash
# Backup original
cp move_detector.py move_detector_original.py

# Replace with improved version
cp move_detector_improved.py move_detector.py
```

### Option 2: Use Directly

```bash
python move_detector_improved.py --video video.mov --holds holds.json
```

### Option 3: Update Pipeline

In `run_climb_pipeline.py`, change import:
```python
from move_detector_improved import detect_and_classify_moves
```

## Expected Results

### Before (Original):
- 19-50+ detected "moves" for a V3-V5 route
- Many false positives from:
  - Hand readjustments
  - Foot micro-movements
  - Swaying while stable
  - Jitter/noise

### After (Improved):
- 8-15 detected moves for a V3-V5 route
- Only real, intentional hold transitions
- Filtered out:
  - Micro-adjustments
  - Oscillations
  - Jitter
  - Minor repositioning

## Tuning Parameters

If you still get too many/few moves, adjust these in the improved file:

```python
# For MORE sensitive (more moves detected):
MIN_STABLE_FRAMES = 5           # Was 8
MIN_HOLD_SWITCH_DIST = 40.0     # Was 60.0
MIN_MOVE_VELOCITY = 10.0        # Was 15.0

# For LESS sensitive (fewer moves detected):
MIN_STABLE_FRAMES = 12          # Was 8
MIN_HOLD_SWITCH_DIST = 80.0     # Was 60.0
MIN_MOVE_VELOCITY = 20.0        # Was 15.0
```

## Testing

Test on your video:

```bash
# Run with improved detector
python move_detector_improved.py \
    --video Vids/climbVid.mov \
    --holds output/hold_positions_auto.json \
    --out-dir output

# Check results
python analyze_moves.py
```

Compare move counts and verify they match actual climbing moves!
