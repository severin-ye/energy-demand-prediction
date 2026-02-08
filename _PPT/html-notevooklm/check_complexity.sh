#!/bin/bash

echo "🚀 检测PPT页面内容复杂度..."
echo "📏 分析每个页面的内容量"
echo ""

for file in [0-9]*.html; do
    if [ -f "$file" ]; then
        # 提取.content区域
        content=$(sed -n '/<div class="content">/,/<\/div>/p' "$file" | head -n -1)
        
        # 统计各类元素
        h2_count=$(echo "$content" | grep -o '<h2' | wc -l)
        h3_count=$(echo "$content" | grep -o '<h3' | wc -l)
        div_count=$(echo "$content" | grep -o '<div' | wc -l)
        p_count=$(echo "$content" | grep -o '<p' | wc -l)
        ul_count=$(echo "$content" | grep -o '<ul' | wc -l)
        table_count=$(echo "$content" | grep -o '<table' | wc -l)
        img_count=$(echo "$content" | grep -o '<img' | wc -l)
        
        # 计算复杂度分数（粗略估算）
        score=$((h2_count * 50 + h3_count * 40 + div_count * 15 + p_count * 20 + ul_count * 30 + table_count * 100 + img_count * 200))
        
        # 状态判断
        if [ $score -gt 1000 ]; then
            status="❌ 高复杂度"
        elif [ $score -gt 700 ]; then
            status="⚠️  中等"
        else
            status="✅ 正常"
        fi
        
        printf "%s %-35s 复杂度: %4d | h2:%d h3:%d div:%d p:%d ul:%d table:%d img:%d\n" \
            "$status" "$file" $score $h2_count $h3_count $div_count $p_count $ul_count $table_count $img_count
    fi
done

echo ""
echo "========================================"
echo "说明:"
echo "• 复杂度 > 1000: 内容很可能超出页面"
echo "• 复杂度 700-1000: 需要检查"
echo "• 复杂度 < 700: 通常正常"
