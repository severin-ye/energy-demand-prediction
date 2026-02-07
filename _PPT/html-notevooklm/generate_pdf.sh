#!/bin/bash

# PDF生成脚本 - 为NotebookLM PPT生成PDF
# 合并所有页面到一个PDF文件

set -e

echo "🚀 开始生成PDF..."

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/pdf_output"
MERGED_HTML="$SCRIPT_DIR/merged_presentation.html"
MERGED_PDF="$SCRIPT_DIR/论文复现PPT-完整版.pdf"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 查找所有HTML文件（按数字排序）
HTML_FILES=($(ls "$SCRIPT_DIR"/[0-9]*.html 2>/dev/null | sort -V))

if [ ${#HTML_FILES[@]} -eq 0 ]; then
    echo "❌ 错误: 找不到任何HTML文件"
    exit 1
fi

echo "📄 找到 ${#HTML_FILES[@]} 个HTML文件"
echo "📝 合并所有页面..."

# 创建合并的HTML文件
cat > "$MERGED_HTML" << 'HEADER'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>能源需求预测的因果可解释AI - 论文复现PPT</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    @page {
      size: 13.333in 7.5in;
      margin: 0;
    }
    
    @media print {
      html, body {
        width: 13.333in;
        margin: 0;
        padding: 0;
        background: white;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
        color-adjust: exact;
      }
      
      .slide {
        margin: 0 !important;
        box-shadow: none !important;
        page-break-after: always;
        page-break-inside: avoid;
        width: 13.333in !important;
        height: 7.5in !important;
      }
      
      .slide:last-child {
        page-break-after: auto;
      }
      
      .footer {
        display: none !important;
      }
    }
  </style>
</head>
<body>
HEADER

# 从每个HTML文件中提取<section>标签内容
for html_file in "${HTML_FILES[@]}"; do
    filename=$(basename "$html_file")
    echo "  ✓ 添加: $filename"
    
    # 提取<section>...</section>之间的内容
    sed -n '/<section/,/<\/section>/p' "$html_file" >> "$MERGED_HTML"
    echo "" >> "$MERGED_HTML"
done

# 添加HTML结束标签
echo "</body>" >> "$MERGED_HTML"
echo "</html>" >> "$MERGED_HTML"

echo "✅ 合并完成: merged_presentation.html"
echo ""

SUCCESS=0

# 检查是否安装了WeasyPrint
if python3 -c "import weasyprint" 2>/dev/null; then
    echo "✓ 使用: WeasyPrint"
    echo "📄 生成完整PDF..."
    
    python3 -c "
from weasyprint import HTML
HTML('$MERGED_HTML').write_pdf('$MERGED_PDF')
" 2>/dev/null && SUCCESS=1

# 如果没有WeasyPrint，尝试使用Chromium
elif command -v chromium &> /dev/null; then
    echo "✓ 使用: Chromium"
    echo "📄 生成完整PDF..."
    
    chromium --headless --disable-gpu --no-sandbox \
        --print-to-pdf="$MERGED_PDF" \
        --print-to-pdf-no-header \
        "file://$MERGED_HTML" 2>/dev/null && SUCCESS=1

# 尝试Chrome
elif command -v google-chrome &> /dev/null; then
    echo "✓ 使用: Google Chrome"
    echo "📄 生成完整PDF..."
    
    google-chrome --headless --disable-gpu --no-sandbox \
        --print-to-pdf="$MERGED_PDF" \
        --print-to-pdf-no-header \
        "file://$MERGED_HTML" 2>/dev/null && SUCCESS=1
fi

if [ $SUCCESS -eq 1 ]; then
    echo ""
    echo "✅ PDF生成完成！"
    echo "📁 输出文件: $MERGED_PDF"
    echo ""
    ls -lh "$MERGED_PDF" | awk '{print "   文件大小: " $5}'
    echo ""
    echo "💡 提示: 已生成包含 ${#HTML_FILES[@]} 页的完整PDF"
else
    echo ""
    echo "❌ 错误: 未找到可用的PDF生成工具"
    echo ""
    echo "请安装以下工具之一:"
    echo "  • WeasyPrint: pip install weasyprint"
    echo "  • Chromium: sudo apt install chromium-browser"
    echo "  • Chrome: 从 Google 下载"
    exit 1
fi
