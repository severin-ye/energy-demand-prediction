#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的CSS PPT溢出检测工具
通过静态分析HTML和CSS估算内容高度
"""

import re
import sys
from pathlib import Path

# 页面配置
SLIDE_HEIGHT_IN = 7.5  # 4:3比例
MAX_CONTENT_HEIGHT_IN = 6.0  # 内容区域最大高度
DPI = 96
MAX_CONTENT_HEIGHT_PX = MAX_CONTENT_HEIGHT_IN * DPI  # 576px

def extract_padding_margin(style_str):
    """从style属性中提取padding和margin"""
    total = 0
    
    # 匹配 padding: 12px 或 padding-top: 12px 等
    for match in re.finditer(r'(?:padding|margin)(?:-(?:top|bottom))?:\s*(\d+)px', style_str):
        total += int(match.group(1))
    
    return total

def estimate_element_height(tag, text, style, depth):
    """估算单个元素的高度"""
    # 基础高度
    base_heights = {
        'h1': 60, 'h2': 50, 'h3': 40, 'h4': 30,
        'p': 25,
        'div': 5,
        'ul': 10, 'ol': 10, 'li': 20,
        'table': 50,
        'tr': 30,
        'td': 25,
        'img': 250,
        'code': 18,
    }
    
    height = base_heights.get(tag, 5)
    
    # 文本内容影响高度
    if text:
        text_len = len(text.strip())
        if text_len > 100:
            height += (text_len // 100) * 15
    
    # 添加padding和margin
    height += extract_padding_margin(style)
    
    # 嵌套深度影响（避免重复计算）
    height = height * (0.8 ** (depth - 1))
    
    return height

def analyze_content_block(html_content):
    """分析.content块中的内容"""
    # 提取.content区域
    content_match = re.search(r'<div class="content"[^>]*>(.*?)</div>\s*<div class="footer"', html_content, re.DOTALL)
    
    if not content_match:
        return {'error': '找不到.content区域'}
    
    content_html = content_match.group(1)
    
    # 分析所有元素
    total_height = 0
    element_count = 0
    
    # 匹配所有开标签
    tag_pattern = r'<(\w+)(?:\s+[^>]*style="([^"]*)")?[^>]*>(.*?)</\1>'
    
    def analyze_recursive(html, depth=0):
        nonlocal total_height, element_count
        
        for match in re.finditer(tag_pattern, html, re.DOTALL):
            tag = match.group(1)
            style = match.group(2) or ''
            inner = match.group(3)
            
            if tag in ['script', 'style']:
                continue
            
            element_count += 1
            
            # 提取纯文本（去除嵌套标签）
            text = re.sub(r'<[^>]+>', '', inner)
            
            # 估算高度
            height = estimate_element_height(tag, text, style, depth)
            total_height += height
            
            # 递归分析嵌套内容（但避免重复计算文本）
            if '<' in inner:
                analyze_recursive(inner, depth + 1)
    
    analyze_recursive(content_html)
    
    # 折算系数（避免过度估计）
    estimated_height = total_height * 0.5
    
    # 计算特殊元素
    img_count = len(re.findall(r'<img\s', content_html))
    table_count = len(re.findall(r'<table\s', content_html))
    
    # 图片通常占用较大空间
    if img_count > 0:
        estimated_height += img_count * 150
    
    overflow = estimated_height - MAX_CONTENT_HEIGHT_PX
    
    return {
        'estimated_height': round(estimated_height, 1),
        'max_allowed': round(MAX_CONTENT_HEIGHT_PX, 1),
        'overflow': round(overflow, 1),
        'percentage': round((estimated_height / MAX_CONTENT_HEIGHT_PX) * 100, 1),
        'element_count': element_count,
        'img_count': img_count,
        'table_count': table_count,
        'status': 'overflow' if overflow > 50 else ('warning' if overflow > -50 else 'ok')
    }

def main():
    print('🔍 CSS PPT 溢出检测工具')
    print(f'📏 页面比例: 16:10 ({SLIDE_HEIGHT_IN}in高)')
    print(f'📐 最大内容高度: {MAX_CONTENT_HEIGHT_PX}px ({MAX_CONTENT_HEIGHT_IN}in)\n')
    
    html_files = sorted(Path('.').glob('[0-9]*.html'))
    
    if not html_files:
        print('❌ 未找到HTML文件')
        return
    
    results = []
    
    for filepath in html_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        result = analyze_content_block(html_content)
        result['file'] = filepath.name
        results.append(result)
        
        if 'error' in result:
            print(f"❌ {filepath.name}")
            print(f"   {result['error']}\n")
            continue
        
        # 状态图标
        if result['status'] == 'ok':
            icon = '✅'
        elif result['status'] == 'warning':
            icon = '⚠️'
        else:
            icon = '❌'
        
        print(f"{icon} {filepath.name}")
        print(f"   估算: {result['estimated_height']}px / {result['max_allowed']}px ({result['percentage']}%)")
        print(f"   元素: {result['element_count']} | 图片: {result['img_count']} | 表格: {result['table_count']}")
        
        if result['overflow'] > 50:
            print(f"   ⚠️  估算超出: {result['overflow']}px")
        elif result['overflow'] > -50:
            print(f"   ⚡ 接近上限 (剩余: {abs(result['overflow'])}px)")
        else:
            print(f"   ✓ 正常 (剩余: {abs(result['overflow'])}px)")
        print()
    
    # 汇总
    print('=' * 70)
    print('📊 检测汇总\n')
    
    overflow = [r for r in results if r.get('status') == 'overflow']
    warning = [r for r in results if r.get('status') == 'warning']
    ok = [r for r in results if r.get('status') == 'ok']
    
    print(f"❌ 可能溢出: {len(overflow)} 个")
    print(f"⚠️  接近上限: {len(warning)} 个")
    print(f"✅ 正常: {len(ok)} 个\n")
    
    if overflow:
        print('⚠️  需要优化的页面:')
        for r in sorted(overflow, key=lambda x: x.get('overflow', 0), reverse=True):
            print(f"   • {r['file']:40s} (估算超出 {r['overflow']}px)")
    
    if warning:
        print('\n💡 建议检查的页面:')
        for r in warning:
            print(f"   • {r['file']:40s} (接近上限, 剩余 {abs(r['overflow'])}px)")
    
    print('\n' + '='  * 70)
    print('📝 说明:')
    print('• 这是静态估算，实际渲染高度可能不同')
    print('• 建议在浏览器中打开check_overflow.html进行精确检测')
    print('• 或者手动打开标记的页面检查')

if __name__ == '__main__':
    main()
