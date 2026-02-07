#!/bin/bash

# PDF生成脚本 - 支持多种工具
# 优先级: Chromium > Firefox > WeasyPrint > wkhtmltopdf

set -e

echo "🚀 开始生成PDF..."

# 获取脚本所在目录（无论从哪里调用都能正确找到文件）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HTML_FILE="$SCRIPT_DIR/presentation.html"
PDF_FILE="$SCRIPT_DIR/能源预测论文复现PPT.pdf"

# 检查HTML文件是否存在
if [ ! -f "$HTML_FILE" ]; then
    echo "❌ 错误: 找不到 presentation.html"
    exit 1
fi

SUCCESS=0

# 方法1: 优先使用虚拟环境中的 WeasyPrint（中文字体最佳）
VENV_PYTHON="$(dirname "$(dirname "$SCRIPT_DIR")")/.venv/bin/python"
if [ -f "$VENV_PYTHON" ]; then
    if "$VENV_PYTHON" -c "import weasyprint" 2>/dev/null; then
        echo "✓ 使用: WeasyPrint (虚拟环境)"
        "$VENV_PYTHON" -c "
from weasyprint import HTML
HTML('$HTML_FILE').write_pdf('$PDF_FILE')
" 2>/dev/null || true
        SUCCESS=1
    fi
fi

# 方法2: 尝试使用系统 WeasyPrint
if [ $SUCCESS -eq 0 ] && python3 -c "import weasyprint" 2>/dev/null; then
    echo "✓ 使用: WeasyPrint (系统)"
    python3 -c "
from weasyprint import HTML
HTML('$HTML_FILE').write_pdf('$PDF_FILE')
" 2>/dev/null || true
    SUCCESS=1
fi

# 方法3: 尝试使用 Chromium/Chrome
if [ $SUCCESS -eq 0 ]; then
    if command -v chromium &> /dev/null; then
        echo "✓ 使用: Chromium"
        chromium --headless --disable-gpu --no-sandbox --disable-setuid-sandbox \
            --no-pdf-header-footer \
            --print-to-pdf="$PDF_FILE" \
            --print-to-pdf-no-header \
            "file://$HTML_FILE" 2>/dev/null || true
        SUCCESS=1
    elif command -v chromium-browser &> /dev/null; then
        echo "✓ 使用: Chromium Browser"
        chromium-browser --headless --disable-gpu --no-sandbox --disable-setuid-sandbox \
            --no-pdf-header-footer \
            --print-to-pdf="$PDF_FILE" \
            --print-to-pdf-no-header \
            "file://$HTML_FILE" 2>/dev/null || true
        SUCCESS=1
    elif command -v google-chrome &> /dev/null; then
        echo "✓ 使用: Google Chrome"
        google-chrome --headless --disable-gpu --no-sandbox --disable-setuid-sandbox \
            --no-pdf-header-footer \
            --print-to-pdf="$PDF_FILE" \
            --print-to-pdf-no-header \
            "file://$HTML_FILE" 2>/dev/null || true
        SUCCESS=1
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
        --orientation Landscape \
        "file://$HTML_FILE" "$PDF_FILE" 2>/dev/null || true
    SUCCESS=1
fi

# 检查是否成功
if [ $SUCCESS -eq 0 ]; then
    echo ""
    echo "❌ 错误: 未找到任何PDF生成工具"
    echo ""
    echo "请选择以下任一方式安装："
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "方式1 - 安装 WeasyPrint (推荐，中文最佳):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  source .venv/bin/activate  # 如果有虚拟环境"
    echo "  pip3 install weasyprint"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "方式2 - 安装 Chromium:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Ubuntu/Debian:  sudo apt install chromium-browser"
    echo "  Fedora:         sudo dnf install chromium"
    echo "  Arch Linux:     sudo pacman -S chromium"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "方式3 - 安装 wkhtmltopdf:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Ubuntu/Debian:  sudo apt install wkhtmltopdf"
    echo "  Fedora:         sudo dnf install wkhtmltopdf"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "方式4 - 手动生成PDF (无需安装):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  1. 在浏览器中打开: file://$(pwd)/presentation.html"
    echo "  2. 按 Ctrl+P 打开打印对话框"
    echo "  3. 目标选择'另存为PDF'"
    echo "  4. 设置: 横向、无边距、无页眉页脚"
    echo "  5. 保存即可"
    echo ""
    exit 1
fi

# 检查PDF是否生成成功
if [ $SUCCESS -eq 1 ]; then
    sleep 1  # 等待文件写入完成
    if [ -f "$PDF_FILE" ]; then
        echo "✅ PDF生成成功!"
        echo "📁 输出文件: $PDF_FILE"
        echo ""
        echo "📊 文件信息:"
        ls -lh "$PDF_FILE"
        echo ""
        echo "💡 查看PDF:"
        echo "  evince '$PDF_FILE'"
        echo "  或: xdg-open '$PDF_FILE'"
    else
        echo "⚠️  工具执行完成，但PDF文件未找到"
        echo "请尝试手动方法或安装其他工具"
        exit 1
    fi
fi
