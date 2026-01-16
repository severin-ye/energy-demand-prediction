#!/bin/bash
# HTML可视化报告快速查看脚本

echo "🎨 电力负荷智能预测 - HTML可视化报告"
echo "=========================================="
echo ""

HTML_DIR="outputs/inference_uci/html_reports"

if [ ! -d "$HTML_DIR" ]; then
    echo "❌ HTML报告目录不存在: $HTML_DIR"
    echo "💡 请先运行推理测试生成报告："
    echo "   python scripts/run_inference_uci.py --n-samples 100"
    exit 1
fi

INDEX_FILE="$HTML_DIR/index.html"

if [ ! -f "$INDEX_FILE" ]; then
    echo "❌ 索引文件不存在: $INDEX_FILE"
    exit 1
fi

echo "✅ 找到HTML报告目录"
echo "📁 位置: $HTML_DIR"
echo ""

# 统计报告数量
NUM_REPORTS=$(ls -1 $HTML_DIR/sample_*.html 2>/dev/null | wc -l)
echo "📊 共生成 $NUM_REPORTS 个样本报告"
echo ""

# 提供查看选项
echo "选择查看方式:"
echo "  1) 在浏览器中打开索引页面（推荐）"
echo "  2) 在浏览器中打开第一个样本"
echo "  3) 显示文件列表"
echo "  4) 启动本地HTTP服务器"
echo ""

read -p "请选择 [1-4]: " choice

case $choice in
    1)
        echo "🌐 正在打开索引页面..."
        if command -v xdg-open &> /dev/null; then
            xdg-open "$INDEX_FILE"
        elif command -v open &> /dev/null; then
            open "$INDEX_FILE"
        else
            echo "💡 请手动打开: $INDEX_FILE"
        fi
        ;;
    2)
        echo "🌐 正在打开第一个样本..."
        FIRST_SAMPLE="$HTML_DIR/sample_000.html"
        if command -v xdg-open &> /dev/null; then
            xdg-open "$FIRST_SAMPLE"
        elif command -v open &> /dev/null; then
            open "$FIRST_SAMPLE"
        else
            echo "💡 请手动打开: $FIRST_SAMPLE"
        fi
        ;;
    3)
        echo "📋 HTML报告列表:"
        ls -lh "$HTML_DIR"/*.html
        ;;
    4)
        echo "🚀 启动本地HTTP服务器..."
        echo "📍 访问地址: http://localhost:8000"
        echo "💡 按 Ctrl+C 停止服务器"
        echo ""
        cd "$HTML_DIR" && python3 -m http.server 8000
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "✅ 完成！"
