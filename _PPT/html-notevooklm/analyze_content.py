#!/usr/bin/env python3
"""
PPT页面内容分析工具
通过解析HTML估算内容复杂度，找出可能溢出的页面
"""

import os
import re
from pathlib import Path
from bs4 import BeautifulSoup

def estimate_element_height(tag, text_length):
    """粗略估算元素高度（像素）"""
    # 基础高度估算规则
    heights = {
        'h1': 60, 'h2': 50, 'h3': 40, 'h4': 30,
        'p': 20 + (text_length // 50) * 15,  # 每50字符增加15px
        'ul': 25, 'ol': 25, 'li': 18,
        'div': 15,
        'table': 40,
        'tr': 30,
        'img': 200,  # 图片通常较大
        'code': 18,
        'pre': 20,
    }
    return heights.get(tag.name, 10)

def analyze_page(filepath):
    """分析单个页面"""
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find(class_='content')
    
    if not content:
        return {
            'file': filepath.name,
            'error': '找不到 .content 元素'
        }
    
    # 递归计算所有元素
    total_height = 0
    element_count = 0
    nested_divs = 0
    has_image = False
    has_table = False
    
    def traverse(element, depth=0):
        nonlocal total_height, element_count, nested_divs, has_image, has_table
        
        if element.name:
            element_count += 1
            text_length = len(element.get_text(strip=True))
            height = estimate_element_height(element, text_length)
            
            # 检查内联样式中的padding和margin
            style = element.get('style', '')
            padding_match = re.findall(r'padding[:-]\s*(\d+)px', style)
            margin_match = re.findall(r'margin[:-]\s*(\d+)px', style)
            
            if padding_match:
                height += sum(int(p) for p in padding_match) * 2
            if margin_match:
                height += sum(int(m) for m in margin_match) * 2
            
            total_height += height
            
            if element.name == 'div':
                nested_divs += 1
            if element.name == 'img':
                has_image = True
            if element.name == 'table':
                has_table = True
            
            # 递归子元素（但不重复计算文本）
            for child in element.children:
                if child.name:  # 只处理标签，不处理文本节点
                    traverse(child, depth + 1)
    
    traverse(content)
    
    # 估算实际高度（考虑嵌套和样式）
    estimated_height = total_height * 0.6  # 折算系数，因为有重复计算
    
    max_allowed = 6.8 * 96  # 652.8px
    overflow = estimated_height - max_allowed
    
    return {
        'file': filepath.name,
        'estimated_height': round(estimated_height, 1),
        'max_allowed': round(max_allowed, 1),
        'overflow': round(overflow, 1),
        'percentage': round((estimated_height / max_allowed) * 100, 1),
        'element_count': element_count,
        'nested_divs': nested_divs,
        'has_image': has_image,
        'has_table': has_table,
        'status': 'overflow' if overflow > 0 else ('warning' if overflow > -50 else 'ok')
    }

def main():
    print('🚀 开始分析PPT页面内容...\n')
    print(f'📏 最大内容高度限制: 652.8px (6.8in)\n')
    
    html_files = sorted(Path('.').glob('[0-9]*.html'))
    
    results = []
    for filepath in html_files:
        result = analyze_page(filepath)
        results.append(result)
        
        if 'error' in result:
            print(f"❌ {result['file']}")
            print(f"   ERROR: {result['error']}\n")
        else:
            status_icon = '✅' if result['status'] == 'ok' else ('⚠️' if result['status'] == 'warning' else '❌')
            print(f"{status_icon} {result['file']}")
            print(f"   估算高度: {result['estimated_height']}px / {result['max_allowed']}px ({result['percentage']}%)")
            print(f"   元素数量: {result['element_count']} | 嵌套div: {result['nested_divs']}")
            
            if result['overflow'] > 0:
                print(f"   ⚠️  估算超出: {result['overflow']}px")
            else:
                print(f"   ✓ 估算剩余: {abs(result['overflow'])}px")
            print()
    
    # 汇总
    print('\n' + '=' * 60)
    print('📋 分析汇总\n')
    
    overflow_pages = [r for r in results if r.get('status') == 'overflow']
    warning_pages = [r for r in results if r.get('status') == 'warning']
    ok_pages = [r for r in results if r.get('status') == 'ok']
    
    print(f"❌ 可能溢出: {len(overflow_pages)} 个")
    print(f"⚠️  接近上限: {len(warning_pages)} 个")
    print(f"✅ 正常范围: {len(ok_pages)} 个\n")
    
    if overflow_pages:
        print('需要关注的页面:')
        for p in sorted(overflow_pages, key=lambda x: x['overflow'], reverse=True):
            print(f"  • {p['file']} (估算超出 {p['overflow']}px)")
    
    print('\n⚠️  注意: 这是基于HTML结构的估算，实际渲染高度可能有差异')
    print('   建议在浏览器中实际检查标记为"可能溢出"的页面')

if __name__ == '__main__':
    main()
