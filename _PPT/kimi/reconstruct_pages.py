#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动重构HTML PPT中有重叠问题的页面
"""

import re
import json

def read_file(filepath):
    """读取文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(filepath, lines):
    """写入文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def reconstruct_page_3(lines, page_start, page_end):
    """重构第3页 - 传统vs可解释框架对比"""
    print("重构第3页...")
    # 图片在右侧 x=480, y=80, w=450, h=250
    # 策略：删除被图片覆盖的右侧空白文本元素
    
    new_lines = []
    in_page = False
    skip_count = 0
    
    for i, line in enumerate(lines):
        if i == page_start:
            in_page = True
        elif i == page_end:
            in_page = False
            
        if in_page:
            # 删除y坐标在80-330范围内，x坐标>480的空文本
            if 'transform="matrix' in line and ('x="528.8' in line or 'x="510' in line):
                # 检查下一行是否是空文本
                if i+1 < len(lines) and '<tspan' in lines[i+1]:
                    if '</tspan></text></g>' in lines[i+1] or 'tspan></text></g>' in lines[i+1]:
                        skip_count += 1
                        continue  # 跳过整个g元素
            
        new_lines.append(line)
    
    print(f"  删除了 {skip_count} 个空文本元素")
    return new_lines

def reconstruct_page_6(lines, page_start, page_end):
    """重构第6页 - 完整模型架构"""
    print("重构第6页...")
    # 图片较大 x=380, y=100, w=550, h=420
    # 删除被覆盖的标题文本
    
    new_lines = []
    skip_count = 0
    
    for i, line in enumerate(lines):
        if page_start <= i <= page_end:
            # 删除y=91.5, x=398的文本（被图片覆盖）
            if 'transform="matrix(1, 0, 0, 1, 398, 91.5' in line:
                skip_count += 1
                continue
        new_lines.append(line)
    
    print(f"  删除了 {skip_count} 个重叠元素")
    return new_lines

def reconstruct_page_9(lines, page_start, page_end):
    """重构第9页 - BN流程图"""
    print("重构第9页...")
    # 图片在左下 x=45, y=295, w=420, h=105
    # 删除被覆盖的文本
    
    new_lines = []
    skip_count = 0
    
    for i, line in enumerate(lines):
        if page_start <= i <= page_end:
            # 删除y=271.5或y=401.2, x接近45或39.8的文本
            if ('transform="matrix(1, 0, 0, 1, 45, 271.5' in line or 
                'transform="matrix(1, 0, 0, 1, 39.8, 401.2' in line):
                skip_count += 1
                continue
        new_lines.append(line)
    
    print(f"  删除了 {skip_count} 个重叠元素")
    return new_lines

def reconstruct_page_12(lines, page_start, page_end):
    """重构第12页 - 预测性能对比（最严重）"""
    print("重构第12页...")
    # 这页问题最严重：表格与图片重叠
    # 策略：删除整个表格，只保留插入的图片和底部统计框
    
    new_lines = []
    skip_mode = False
    skip_count = 0
    kept_lines = 0
    
    for i, line in enumerate(lines):
        if i < page_start or i > page_end:
            new_lines.append(line)
            continue
            
        # 在页面范围内
        # 保留页面标记
        if '<!-- 第 12 页' in line or '<div class="slide"' in line or '</svg></div>' in line:
            new_lines.append(line)
            kept_lines += 1
            continue
        
        # 保留foreignObject（插入的图片）
        if '<foreignObject x="34.5" y="320"' in line or '<!-- 插入' in line:
            skip_mode = False
            new_lines.append(line)
            kept_lines += 1
            continue
        
        # 保留foreignObject的内容
        if skip_mode == False and ('</foreignObject>' in line or 
                                     '<div xmlns=' in line or 
                                     '<img src="images/7-' in line or
                                     '</div>' in line):
            new_lines.append(line)
            kept_lines += 1
            if '</foreignObject>' in line:
                skip_mode = False
            continue
        
        # 保留底部的三个统计框（从y=424.5开始）
        if 'y="424.5' in line or ('transform="matrix(1, 0, 0, 1, 3' in line and ', 437.25)' in line):
            skip_mode = False
            new_lines.append(line)
            kept_lines += 1
            continue
        
        if skip_mode == False and (', 470.25)' in line or ', 488.25)' in line or 
                                     '34.84%' in line or '13.63%' in line or 
                                     'UCI数据集' in line or 'REFIT数据集' in line or
                                     '统计显著' in line or 't-test验证' in line or
                                     'S-CNN-LSTM的平均提升' in line):
            new_lines.append(line)
            kept_lines += 1
            continue
        
        # 删除表格相关内容
        if ('<g transform=' in line or '<path ' in line or '<text ' in line or 
            '<tspan' in line or '</g>' in line or '</text>' in line or '</tspan>' in line):
            # 检查是否是表格内容（y坐标在80-380之间）
            if '</g>' not in line or skip_mode:
                skip_count += 1
                skip_mode = True
                continue
    
    print(f"  删除了约 {skip_count} 个表格元素")
    print(f"  保留了 {kept_lines} 行关键内容")
    return new_lines

def reconstruct_page_15(lines, page_start, page_end):
    """重构第15页 - BN推理案例"""
    print("重构第15页...")
    # 图片在左上 x=35, y=115, w=285, h=200
    # 删除被图片覆盖的左侧文本
    
    new_lines = []
    skip_count = 0
    
    for i, line in enumerate(lines):
        if page_start <= i <= page_end:
            # 删除与图片重叠的文本元素
            if 'transform="matrix(1, 0, 0, 1, 39, 94.5' in line or \
               'transform="matrix(1, 0, 0, 1, 72, 96' in line or \
               ('transform="matrix(1, 0, 0, 1, 42, ' in line and 
                any(y in line for y in ['226.5', '247.5', '277.5', '285', '292.5', '306', '324'])) or \
               'transform="matrix(1, 0, 0, 1, 48, 277.5' in line or \
               'transform="matrix(1, 0, 0, 1, 48, 292.5' in line:
                skip_count += 1
                continue
        new_lines.append(line)
    
    print(f"  删除了 {skip_count} 个重叠元素")
    return new_lines

def main():
    """主函数"""
    filepath = 'CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction.html'
    output_file = 'CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction_重构.html'
    
    print("=" * 80)
    print("开始自动重构HTML PPT")
    print("=" * 80)
    
    # 读取文件
    lines = read_file(filepath)
    print(f"\n读取文件: {filepath}")
    print(f"总行数: {len(lines)}")
    
    # 找到页面位置
    page_positions = {}
    current_page = None
    
    for i, line in enumerate(lines):
        if '<!-- 第' in line and '页' in line:
            match = re.search(r'第\s*(\d+)\s*页', line)
            if match:
                page_num = int(match.group(1))
                current_page = page_num
                page_positions[page_num] = {'start': i}
        elif current_page and '</svg></div>' in line:
            if current_page in page_positions:
                page_positions[current_page]['end'] = i
                current_page = None
    
    print(f"\n识别到 {len(page_positions)} 个页面")
    
    # 重构各个页面
    print("\n开始重构...")
    
    # 第3页
    if 3 in page_positions:
        lines = reconstruct_page_3(lines, page_positions[3]['start'], page_positions[3]['end'])
    
    # 第6页
    if 6 in page_positions:
        lines = reconstruct_page_6(lines, page_positions[6]['start'], page_positions[6]['end'])
    
    # 第9页
    if 9 in page_positions:
        lines = reconstruct_page_9(lines, page_positions[9]['start'], page_positions[9]['end'])
    
    # 第12页（最重要）
    if 12 in page_positions:
        lines = reconstruct_page_12(lines, page_positions[12]['start'], page_positions[12]['end'])
    
    # 第15页
    if 15 in page_positions:
        lines = reconstruct_page_15(lines, page_positions[15]['start'], page_positions[15]['end'])
    
    # 写入文件
    write_file(output_file, lines)
    print(f"\n重构完成！")
    print(f"输出文件: {output_file}")
    print(f"新文件行数: {len(lines)}")
    
    print("\n" + "=" * 80)
    print("建议：")
    print("  1. 检查重构后的HTML文件")
    print("  2. 运行检测脚本验证重叠是否解决")
    print("  3. 生成新的PDF查看效果")
    print("=" * 80)

if __name__ == '__main__':
    main()
