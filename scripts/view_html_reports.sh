#!/bin/bash
# Quick view script for HTML visualization reports

echo "🎨 Intelligent Power Load Prediction - HTML Visualization Reports"
echo "=========================================="
echo ""

HTML_DIR="outputs/inference_uci/html_reports"

if [ ! -d "$HTML_DIR" ]; then
    echo "❌ HTML report directory does not exist: $HTML_DIR"
    echo "💡 Please run inference testing first to generate reports:"
    echo "   python scripts/run_inference_uci.py --n-samples 100"
    exit 1
fi

INDEX_FILE="$HTML_DIR/index.html"

if [ ! -f "$INDEX_FILE" ]; then
    echo "❌ Index file does not exist: $INDEX_FILE"
    exit 1
fi

echo "✅ Found HTML report directory"
echo "📁 Location: $HTML_DIR"
echo ""

# Count number of reports
NUM_REPORTS=$(ls -1 $HTML_DIR/sample_*.html 2>/dev/null | wc -l)
echo "📊 Generated $NUM_REPORTS sample reports in total"
echo ""

# Provide viewing options
echo "Choose viewing method:"
echo "  1) Open index page in browser (recommended)"
echo "  2) Open first sample in browser"
echo "  3) Show file list"
echo "  4) Start local HTTP server"
echo ""

read -p "Please select [1-4]: " choice

case $choice in
    1)
        echo "🌐 Opening index page..."
        if command -v xdg-open &> /dev/null; then
            xdg-open "$INDEX_FILE"
        elif command -v open &> /dev/null; then
            open "$INDEX_FILE"
        else
            echo "💡 Please open manually: $INDEX_FILE"
        fi
        ;;
    2)
        echo "🌐 Opening first sample..."
        FIRST_SAMPLE="$HTML_DIR/sample_000.html"
        if command -v xdg-open &> /dev/null; then
            xdg-open "$FIRST_SAMPLE"
        elif command -v open &> /dev/null; then
            open "$FIRST_SAMPLE"
        else
            echo "💡 Please open manually: $FIRST_SAMPLE"
        fi
        ;;
    3)
        echo "📋 HTML report list:"
        ls -lh "$HTML_DIR"/*.html
        ;;
    4)
        echo "🚀 Starting local HTTP server..."
        echo "📍 Access URL: http://localhost:8000"
        echo "💡 Press Ctrl+C to stop server"
        echo ""
        cd "$HTML_DIR" && python3 -m http.server 8000
        ;;
    *)
        echo "❌ Invalid selection"
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
