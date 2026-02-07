#!/bin/bash

# PDF导出脚本
# 将修复后的HTML演示文稿导出为PDF

set -e

echo "🚀 开始导出PDF..."

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HTML_FILE="$SCRIPT_DIR/CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction.html"
PDF_FILE="$SCRIPT_DIR/因果可解释AI论文复现PPT.pdf"

# 检查HTML文件是否存在
if [ ! -f "$HTML_FILE" ]; then
    echo "❌ 错误: 找不到HTML文件"
    exit 1
fi

SUCCESS=0

# 方法1: 尝试使用 Chromium/Chrome (最佳质量)
if [ $SUCCESS -eq 0 ]; then
    if command -v chromium &> /dev/null; then
        echo "✓ 使用: Chromium"
        chromium --headless --disable-gpu --no-sandbox --disable-setuid-sandbox \
            --no-pdf-header-footer \
            --print-to-pdf="$PDF_FILE" \
            --print-to-pdf-no-header \
            "file://$HTML_FILE" 2>/dev/null && SUCCESS=1
    elif command -v chromium-browser &> /dev/null; then
        echo "✓ 使用: Chromium Browser"
        chromium-browser --headless --disable-gpu --no-sandbox --disable-setuid-sandbox \
            --no-pdf-header-footer \
            --print-to-pdf="$PDF_FILE" \
            --print-to-pdf-no-header \
            "file://$HTML_FILE" 2>/dev/null && SUCCESS=1
    elif command -v google-chrome &> /dev/null; then
        echo "✓ 使用: Google Chrome"
        google-chrome --headless --disable-gpu --no-sandbox --disable-setuid-sandbox \
            --no-pdf-header-footer \
            --print-to-pdf="$PDF_FILE" \
            --print-to-pdf-no-header \
            "file://$HTML_FILE" 2>/dev/null && SUCCESS=1
    fi
fi

# 方法2: 尝试使用 Firefox
if [ $SUCCESS -eq 0 ] && command -v firefox &> /dev/null; then
    echo "✓ 使用: Firefox"
    firefox --headless --screenshot "$PDF_FILE" "file://$HTML_FILE" 2>/dev/null && SUCCESS=1
fi

# 方法3: 尝试使用 WeasyPrint
if [ $SUCCESS -eq 0 ]; then
    if python3 -c "import weasyprint" 2>/dev/null; then
        echo "✓ 使用: WeasyPrint"
        python3 -c "
from weasyprint import HTML
HTML('$HTML_FILE').write_pdf('$PDF_FILE')
" 2>/dev/null && SUCCESS=1
    fi
fi

# 方法4: 尝试使用 wkhtmltopdf
if [ $SUCCESS -eq 0 ] && command -v wkhtmltopdf &> /dev/null; then
    echo "✓ 使用: wkhtmltopdf"
    wkhtmltopdf --page-size "Custom" \
        --page-width "13.333in" \
        --page-height "7.5in" \
        --margin-top 0 \
        --margin-bottom 0 \
        --margin-left 0 \
        --margin-right 0 \
        --disable-smart-shrinking \
        "$HTML_FILE" "$PDF_FILE" 2>/dev/null && SUCCESS=1
fi

# 检查生成结果
if [ -f "$PDF_FILE" ]; then
    FILE_SIZE=$(du -h "$PDF_FILE" | cut -f1)
    echo ""
    echo "✅ PDF生成成功!"
    echo "📁 文件位置: $PDF_FILE"
    echo "📊 文件大小: $FILE_SIZE"
    echo ""
    echo "提示: 可以使用以下命令打开PDF:"
    echo "  xdg-open \"$PDF_FILE\""
else
    echo ""
    echo "❌ PDF生成失败"
    echo ""
    echo "请尝试以下方法之一:"
    echo "1. 安装 Chromium:  sudo apt install chromium-browser"
    echo "2. 安装 WeasyPrint: pip install weasyprint"
    echo "3. 在浏览器中打开HTML并使用打印功能(Ctrl+P)手动导出"
    echo ""
    exit 1
fi
