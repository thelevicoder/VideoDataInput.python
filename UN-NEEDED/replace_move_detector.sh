#!/bin/bash
# replace_move_detector.sh
#
# Quick script to backup original and install improved move detector

echo "================================================"
echo "MOVE DETECTOR REPLACEMENT SCRIPT"
echo "================================================"
echo ""

# Check if original exists
if [ ! -f "move_detector.py" ]; then
    echo "❌ Error: move_detector.py not found in current directory"
    echo "   Run this script from your project root"
    exit 1
fi

# Backup original
echo "📦 Backing up original move_detector.py..."
cp move_detector.py move_detector_original.py
echo "   ✅ Saved as move_detector_original.py"
echo ""

# Install improved version
echo "🔧 Installing improved move detector..."
cp move_detector_improved.py move_detector.py
echo "   ✅ Replaced move_detector.py"
echo ""

echo "================================================"
echo "✅ INSTALLATION COMPLETE"
echo "================================================"
echo ""
echo "The improved move detector has been installed!"
echo ""
echo "Key improvements:"
echo "  • Stricter stability requirements (8 frames vs 3)"
echo "  • Larger minimum movement distance (60px vs 8px)"
echo "  • Velocity-based detection (15px/frame minimum)"
echo "  • Oscillation filtering"
echo "  • Resolution-adaptive thresholds"
echo ""
echo "To restore the original:"
echo "  cp move_detector_original.py move_detector.py"
echo ""
echo "To test:"
echo "  python run_climb_pipeline.py --video your_video.mov"
echo ""
