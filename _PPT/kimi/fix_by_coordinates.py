#!/usr/bin/env python3
"""
基于坐标删除重叠元素 - 精确版
"""

import re

def parse_transform_matrix(transform_str):
    """解析 transform="matrix(...)" 获取坐标"""
    match = re.search(r'matrix\([^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*([^,]+),\s*([^)]+)\)', transform_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def check_overlap(text_x, text_y, img_x, img_y, img_w, img_h, margin=5):
    """检查文本元素是否与图片重叠"""
    # 简化:只检查文本起始点是否在图片范围内
    return (img_x - margin <= text_x <= img_x + img_w + margin and 
            img_y - margin <= text_y <= img_y + img_h + margin)

# 图片信息 (从检测报告)
images_info = {
    3: {'x': 480, 'y': 80, 'w': 450, 'h': 250},
    6: {'x': 380, 'y': 100, 'w': 550, 'h': 420},
    9: {'x': 45, 'y': 295, 'w': 420, 'h': 105},
    15: {'x': 35, 'y': 115, 'w': 285, 'h': 200},
}

with open('CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction_clean.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找每个页面范围
def get_page_range(page_num):
    start = None
    end = None
    for i, line in enumerate(lines):
        if f'<!-- 第 {page_num} 页 -->' in line:
            start = i
        elif start and f'<!-- 第 {page_num+1} 页 -->' in line:
            end = i
            break
    return start, end

# 标记要删除的行
lines_to_delete = set()

for page_num, img_info in images_info.items():
    page_start, page_end = get_page_range(page_num)
    if not page_start:
        continue
    
    print(f"\n处理第{page_num}页 (lines {page_start}-{page_end})...")
    img_x, img_y, img_w, img_h = img_info['x'], img_info['y'], img_info['w'], img_info['h']
    print(f"  图片区域: x={img_x}, y={img_y}, w={img_w}, h={img_h}")
    
    deleted_count = 0
    i = page_start
    while i < page_end:
        line = lines[i]
        
        # 查找 <g transform="matrix"> 元素
        if '<g transform="matrix' in line:
            # 解析坐标
            x, y = parse_transform_matrix(line)
            if x is not None and y is not None:
                # 检查是否重叠
                if check_overlap(x, y, img_x, img_y, img_w, img_h):
                    # 找到这个 <g> 块的结束
                    block_end = i
                    for j in range(i+1, min(i+20, page_end)):
                        if '</g>' in lines[j]:
                            block_end = j
                            break
                    
                    # 标记整个块删除
                    for k in range(i, block_end+1):
                        lines_to_delete.add(k)
                    
                    deleted_count += 1
                    print(f"    删除 ({x:.1f}, {y:.1f}) lines {i}-{block_end}")
                    i = block_end + 1
                    continue
        
        i += 1
    
    print(f"  共删除 {deleted_count} 个重叠元素")

# 生成新文件
new_lines = [line for i, line in enumerate(lines) if i not in lines_to_delete]

with open('CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction_final.html', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"\n" + "="*60)
print(f"原始: {len(lines)} 行")
print(f"删除: {len(lines_to_delete)} 行")
print(f"最终: {len(new_lines)} 行")
print(f"输出: CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction_final.html")
print("="*60)
