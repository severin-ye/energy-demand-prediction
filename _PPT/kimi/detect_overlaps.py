#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测HTML PPT中的重叠问题并生成重构建议
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Element:
    """页面元素"""
    type: str  # 'text', 'image', 'rect', etc.
    x: float
    y: float
    width: float
    height: float
    line_num: int
    content: str = ""

@dataclass
class Page:
    """页面信息"""
    number: int
    start_line: int
    end_line: int
    elements: List[Element]
    has_image: bool = False

def extract_number(s: str) -> float:
    """提取数字"""
    match = re.search(r'[-+]?\d*\.?\d+', s)
    return float(match.group()) if match else 0.0

def parse_transform(transform: str) -> Tuple[float, float]:
    """解析transform matrix获取x, y坐标"""
    match = re.search(r'matrix\([^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\)', transform)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    match = re.search(r'translate\(([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\)', transform)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    return 0.0, 0.0

def elements_overlap(e1: Element, e2: Element, margin: float = 5.0) -> bool:
    """检测两个元素是否重叠（加入margin容错）"""
    # 计算元素边界
    e1_left = e1.x - margin
    e1_right = e1.x + e1.width + margin
    e1_top = e1.y - margin
    e1_bottom = e1.y + e1.height + margin
    
    e2_left = e2.x - margin
    e2_right = e2.x + e2.width + margin
    e2_top = e2.y - margin
    e2_bottom = e2.y + e2.height + margin
    
    # 检测是否重叠
    return not (e1_right < e2_left or e1_left > e2_right or 
                e1_bottom < e2_top or e1_top > e2_bottom)

def analyze_html(filepath: str) -> Dict:
    """分析HTML文件，检测重叠"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    pages = []
    current_page = None
    page_num = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检测页面开始
        if '<!-- 第' in line and '页' in line:
            page_match = re.search(r'第\s*(\d+)\s*页', line)
            if page_match:
                page_num = int(page_match.group(1))
                current_page = Page(number=page_num, start_line=i, end_line=i, elements=[])
                pages.append(current_page)
        
        # 检测页面结束
        if current_page and '</svg></div>' in line:
            current_page.end_line = i
            
            # 解析该页面的元素
            page_content = ''.join(lines[current_page.start_line:current_page.end_line + 1])
            
            # 查找foreignObject（图片）
            for match in re.finditer(r'<foreignObject\s+x="([\d.]+)"\s+y="([\d.]+)"\s+width="([\d.]+)"\s+height="([\d.]+)"', page_content):
                x, y, w, h = map(float, match.groups())
                # 找到对应的行号
                offset = match.start()
                line_num = current_page.start_line + page_content[:offset].count('\n')
                current_page.elements.append(Element('image', x, y, w, h, line_num))
                current_page.has_image = True
            
            # 查找文本元素 <g transform=...><text>
            for match in re.finditer(r'<g\s+transform="([^"]+)"><text[^>]*>([^<]*)', page_content):
                transform_str = match.group(1)
                text_content = match.group(2)
                tx, ty = parse_transform(transform_str)
                
                # 估算文本大小（粗略估计）
                text_width = len(text_content) * 8  # 假设每个字符8px
                text_height = 20  # 假设行高20px
                
                offset = match.start()
                line_num = current_page.start_line + page_content[:offset].count('\n')
                current_page.elements.append(Element('text', tx, ty, text_width, text_height, line_num, text_content))
            
            # 查找矩形元素
            for match in re.finditer(r'<rect[^>]+x="([\d.]+)"[^>]+y="([\d.]+)"[^>]+width="([\d.]+)"[^>]+height="([\d.]+)"', page_content):
                x, y, w, h = map(float, match.groups())
                offset = match.start()
                line_num = current_page.start_line + page_content[:offset].count('\n')
                current_page.elements.append(Element('rect', x, y, w, h, line_num))
        
        i += 1
    
    # 检测重叠
    results = {
        'pages': pages,
        'overlapping_pages': [],
        'summary': {}
    }
    
    for page in pages:
        if not page.has_image:
            continue
        
        overlaps = []
        images = [e for e in page.elements if e.type == 'image']
        texts = [e for e in page.elements if e.type == 'text']
        
        # 检测图片与文本的重叠
        for img in images:
            for txt in texts:
                if elements_overlap(img, txt):
                    overlaps.append({
                        'image': img,
                        'text': txt,
                        'type': 'image-text'
                    })
        
        if overlaps:
            results['overlapping_pages'].append({
                'page_number': page.number,
                'overlaps': overlaps,
                'page_info': page
            })
    
    # 生成摘要
    results['summary'] = {
        'total_pages': len(pages),
        'pages_with_images': len([p for p in pages if p.has_image]),
        'pages_with_overlaps': len(results['overlapping_pages']),
        'overlap_count': sum(len(p['overlaps']) for p in results['overlapping_pages'])
    }
    
    return results

def print_report(results: Dict):
    """打印检测报告"""
    print("=" * 80)
    print("HTML PPT 重叠检测报告")
    print("=" * 80)
    print(f"\n总页数: {results['summary']['total_pages']}")
    print(f"包含图片的页面数: {results['summary']['pages_with_images']}")
    print(f"存在重叠的页面数: {results['summary']['pages_with_overlaps']}")
    print(f"重叠总数: {results['summary']['overlap_count']}")
    
    if results['overlapping_pages']:
        print("\n" + "=" * 80)
        print("详细重叠信息:")
        print("=" * 80)
        
        for page_info in results['overlapping_pages']:
            page_num = page_info['page_number']
            overlaps = page_info['overlaps']
            
            print(f"\n【第 {page_num} 页】 发现 {len(overlaps)} 处重叠:")
            print("-" * 80)
            
            for idx, overlap in enumerate(overlaps, 1):
                img = overlap['image']
                txt = overlap['text']
                print(f"\n  重叠 {idx}:")
                print(f"    图片位置: x={img.x:.1f}, y={img.y:.1f}, w={img.width:.1f}, h={img.height:.1f}")
                print(f"    图片行号: {img.line_num}")
                print(f"    文本位置: x={txt.x:.1f}, y={txt.y:.1f}")
                print(f"    文本内容: {txt.content[:50]}..." if len(txt.content) > 50 else f"    文本内容: {txt.content}")
                print(f"    文本行号: {txt.line_num}")
    
    print("\n" + "=" * 80)
    print("建议:")
    print("=" * 80)
    
    if results['overlapping_pages']:
        print("\n需要重构的页面:")
        for page_info in results['overlapping_pages']:
            print(f"  - 第 {page_info['page_number']} 页 ({len(page_info['overlaps'])} 处重叠)")
        
        print("\n重构策略:")
        print("  1. 调整图片位置（移动到不重叠的区域）")
        print("  2. 调整图片大小（缩小以适应空间）")
        print("  3. 删除或移动重叠的文本元素")
        print("  4. 重新布局整个页面")
    else:
        print("\n✓ 未检测到重叠问题！")

if __name__ == '__main__':
    filepath = 'CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction.html'
    results = analyze_html(filepath)
    print_report(results)
    
    # 保存结果到JSON
    import json
    with open('overlap_report.json', 'w', encoding='utf-8') as f:
        # 转换dataclass为dict
        output = {
            'summary': results['summary'],
            'overlapping_pages': [
                {
                    'page_number': p['page_number'],
                    'overlap_count': len(p['overlaps'])
                }
                for p in results['overlapping_pages']
            ]
        }
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细报告已保存到: overlap_report.json")
