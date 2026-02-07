#!/usr/bin/env python3
"""
智能修复页面重叠 - 版本2
通过直接操作行号来删除重叠元素
"""

import re

def fix_page_12(html_lines):
    """
    第12页:删除大部分表格,只保留标题和最后一行(我们的模型),图片上移
    
    表格结构分析:
    - 表格开始: ~line 759
    - 表格表头: ~ line 759-843  
    - 数据行: ~line 844-921
    - 图片: line 924 (y=320)
    - 统计框: line 927+ (y=424)
    
    重叠原因:表格底部 y≈400, 图片顶部 y=320
    
    解决方案:
    1. 保留表格框架 (lines 759-843)
    2. 只保留最后一行数据 (P-CNN-LSTM-Att)
    3. 删除其他数据行
    4. 图片位置不变
    """
    
    # 找到第12页的起始和结束位置
    page_12_start = None
    page_12_end = None
    
    for i, line in enumerate(html_lines):
        if '<!-- 第 12 页 -->' in line:
            page_12_start = i
        elif page_12_start and '<!-- 第 13 页 -->' in line:
            page_12_end = i
            break
    
    if not page_12_start:
        print("⚠️  未找到第12页")
        return html_lines
    
    print(f"📍 第12页位置: lines {page_12_start} - {page_12_end}")
    
    # 在第12页范围内查找需要删除的行
    # 策略:找到表格数据行,只保留最后一行(P-CNN-LSTM-Att)
    
    table_data_start = None
    our_model_line_start = None
    our_model_line_end = None
    image_line = None
    
    for i in range(page_12_start, page_12_end):
        line = html_lines[i]
        
        # 找到图片位置 (图片在 foreignObject 内部几行)
        if '7-预测性能与MSE对比结果.png' in line:
            image_line = i
            
        # 找到我们模型的那一行
        if 'P-CNN-LSTM-Att (Ours)' in line:
            our_model_line_start = i
            # 向下查找,找到这一行的结束 (通过找到下一个 <g transform)
            for j in range(i+1, min(i+30, page_12_end)):
                # 找到下一个不是我们模型数据的行
                if '</g>' in html_lines[j] and '<g transform' in html_lines[j+1] if j+1 < len(html_lines) else False:
                    # 检查下一行是否还是我们模型的数据
                    next_content = ''.join(html_lines[j+1:j+5])
                    if '0.01433' not in next_content and '0.05541' not in next_content and '0.03628' not in next_content:
                        our_model_line_end = j
                        break
            if not our_model_line_end:
                our_model_line_end = i + 10  # 默认10行
        
        # 找到Ensemble开始位置 (表格数据的开始)
        if table_data_start is None and 'Ensemble' in line:
            table_data_start = i
    
    print(f"  表格数据开始: line {table_data_start}")
    print(f"  我们的模型: lines {our_model_line_start} - {our_model_line_end}")
    print(f"  图片位置: line {image_line}")
    
    if not all([table_data_start, our_model_line_start, image_line]):
        print("⚠️  未能找到所有关键元素")
        return html_lines
    
    # 删除策略:删除 table_data_start 到 our_model_line_start 之间的所有行
    # 只保留表头和我们的模型行
    
    lines_to_delete = list(range(table_data_start, our_model_line_start))
    
    print(f"  将删除 {len(lines_to_delete)} 行表格数据")
    
    # 创建新的HTML,跳过要删除的行
    new_lines = []
    for i, line in enumerate(html_lines):
        if i not in lines_to_delete:
            new_lines.append(line)
    
    print(f"✅ 第12页重构完成: 删除了 {len(lines_to_delete)} 行")
    
    return new_lines


def fix_page_3(html_lines):
    """
    第3页:删除图片下方的空文本元素
    
    图片位置: x=480, y=80, w=450, h=250 (到 y=330)
    重叠文本坐标: (528.8, 85.5), (510, 285), (510, 305), (510, 323), (510, 325.5)
    
    这些都是空的 <text> 元素,直接删除
    """
    
    # 找到第3页
    page_3_start = None
    page_3_end = None
    
    for i, line in enumerate(html_lines):
        if '<!-- 第 3 页 -->' in line:
            page_3_start = i
        elif page_3_start and '<!-- 第 4 页 -->' in line:
            page_3_end = i
            break
    
    if not page_3_start:
        return html_lines
    
    # 在第3页范围内查找空文本元素
    # 这些元素特征: <g transform="matrix..."><text ... /></g> 但 text 内容为空或只有空格
    
    lines_to_delete = []
    
    i = page_3_start
    while i < page_3_end:
        line = html_lines[i]
        
        # 检查是否是 <g transform> 行
        if '<g transform="matrix' in line:
            # 检查接下来几行是否构成一个空元素
            block_end = i
            for j in range(i+1, min(i+5, page_3_end)):
                if '</g>' in html_lines[j]:
                    block_end = j
                    break
            
            # 提取这个块的内容
            block_content = ''.join(html_lines[i:block_end+1])
            
            # 检查是否是空文本 (只有 <text> 标签但没有 <tspan> 或内容)
            if '<text' in block_content and '</text>' in block_content:
                # 如果没有 tspan 或 tspan 是空的
                if '<tspan' not in block_content or (
                    '<tspan' in block_content and 
                    re.search(r'<tspan[^>]*>\s*</tspan>', block_content)
                ):
                    # 这是一个空文本元素,标记删除
                    for k in range(i, block_end+1):
                        lines_to_delete.append(k)
                    i = block_end + 1
                    continue
        
        i += 1
    
    if lines_to_delete:
        new_lines = [line for i, line in enumerate(html_lines) if i not in lines_to_delete]
        print(f"✅ 第3页:删除了 {len(set(lines_to_delete))} 个空文本元素")
        return new_lines
    
    return html_lines


def fix_page_6(html_lines):
    """
    第6页:删除被图片覆盖的文字
    图片: x=380, y=100, w=550, h=420
    重叠: (398, 91.5)
    """
    
    page_6_start = None
    page_6_end = None
    
    for i, line in enumerate(html_lines):
        if '<!-- 第 6 页 -->' in line:
            page_6_start = i
        elif page_6_start and '<!-- 第 7 页 -->' in line:
            page_6_end = i
            break
    
    if not page_6_start:
        return html_lines
    
    lines_to_delete = []
    
    # 查找 transform 包含 398 的元素
    for i in range(page_6_start, page_6_end):
        if 'transform="matrix' in html_lines[i] and ', 398,' in html_lines[i]:
            # 找到这个元素的结束
            for j in range(i+1, min(i+10, page_6_end)):
                if '</g>' in html_lines[j]:
                    for k in range(i, j+1):
                        lines_to_delete.append(k)
                    break
    
    if lines_to_delete:
        new_lines = [line for i, line in enumerate(html_lines) if i not in lines_to_delete]
        print(f"✅ 第6页:删除了 {len(set(lines_to_delete))} 行重叠文本")
        return new_lines
    
    return html_lines


def fix_page_9(html_lines):
    """
    第9页:删除BN工作流图周围的重叠文本
    图片: x=45, y=295, w=420, h=105
    重叠: (45, 271.5), (39.8, 401.2)
    """
    
    page_9_start = None
    page_9_end = None
    
    for i, line in enumerate(html_lines):
        if '<!-- 第 9 页 -->' in line:
            page_9_start = i
        elif page_9_start and '<!-- 第 10 页 -->' in line:
            page_9_end = i
            break
    
    if not page_9_start:
        return html_lines
    
    lines_to_delete = []
    
    # 查找包含这些坐标的元素
    for i in range(page_9_start, page_9_end):
        line = html_lines[i]
        if 'transform="matrix' in line:
            # 检查坐标
            if ', 45, 271.5)' in line or ', 39.8, 401.2)' in line:
                # 删除这个元素
                for j in range(i+1, min(i+10, page_9_end)):
                    if '</g>' in html_lines[j]:
                        for k in range(i, j+1):
                            lines_to_delete.append(k)
                        break
    
    if lines_to_delete:
        new_lines = [line for i, line in enumerate(html_lines) if i not in lines_to_delete]
        print(f"✅ 第9页:删除了 {len(set(lines_to_delete))} 行重叠文本")
        return new_lines
    
    return html_lines


def fix_page_15(html_lines):
    """
    第15页:删除BN推理案例图左侧的重叠文本
    图片: x=35, y=115, w=285, h=200
    重叠的Y坐标: 94.5, 96, 226.5, 247.5, 277.5, 285, 292.5, 306, 324
    """
    
    page_15_start = None
    page_15_end = None
    
    for i, line in enumerate(html_lines):
        if '<!-- 第 15 页 -->' in line:
            page_15_start = i
        elif page_15_start and '<!-- 第 16 页 -->' in line:
            page_15_end = i
            break
    
    if not page_15_start:
        return html_lines
    
    # 重叠的Y坐标范围
    overlap_y_coords = ['94.5', '96', '226.5', '247.5', '277.5', '285', '292.5', '306', '324']
    
    lines_to_delete = []
    
    for i in range(page_15_start, page_15_end):
        line = html_lines[i]
        if 'transform="matrix' in line:
            # 检查是否包含重叠的Y坐标
            for y_coord in overlap_y_coords:
                if f', {y_coord})' in line:
                    # 删除这个元素
                    for j in range(i+1, min(i+10, page_15_end)):
                        if '</g>' in html_lines[j]:
                            for k in range(i, j+1):
                                lines_to_delete.append(k)
                            break
                    break
    
    if lines_to_delete:
        new_lines = [line for i, line in enumerate(html_lines) if i not in lines_to_delete]
        print(f"✅ 第15页:删除了 {len(set(lines_to_delete))} 行重叠文本")
        return new_lines
    
    return html_lines


def main():
    input_file = 'CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction.html'
    output_file = 'CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction_修复版.html'
    
    print("=" * 60)
    print("智能修复页面重叠 - 版本2")
    print("=" * 60)
    
    # 读取HTML
    with open(input_file, 'r', encoding='utf-8') as f:
        html_lines = f.readlines()
    
    print(f"\n原始文件: {len(html_lines)} 行\n")
    
    # 依次修复每个页面
    html_lines = fix_page_3(html_lines)
    html_lines = fix_page_6(html_lines)
    html_lines = fix_page_9(html_lines)
    html_lines = fix_page_12(html_lines)
    html_lines = fix_page_15(html_lines)
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(html_lines)
    
    print(f"\n新文件: {len(html_lines)} 行")
    print(f"输出文件: {output_file}")
    print("\n" + "=" * 60)
    print("✅ 修复完成!")
    print("=" * 60)
    
    print("\n下一步:")
    print("  1. 运行检测: python3 detect_overlaps.py", output_file)
    print("  2. 检查是否还有重叠")


if __name__ == '__main__':
    main()
