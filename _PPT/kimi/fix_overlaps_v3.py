#!/usr/bin/env python3
"""
智能修复页面重叠 - 版本3
更激进的重构策略
"""

def fix_page_12_aggressive(html_lines):
    """
    第12页完全重新布局:
    1. 保留页面标题
    2. 删除整个表格
    3. 保留图片(上移到 y=100)
    4. 保留底部三个统计框
    """
    
    # 找到第12页
    page_start = None
    page_end = None
    
    for i, line in enumerate(html_lines):
        if '<!-- 第 12 页 -->' in line:
            page_start = i
        elif page_start and '<!-- 第 13 页 -->' in line:
            page_end = i
            break
    
    if not page_start:
        return html_lines
    
    print(f"📍 第12页: lines {page_start} - {page_end}")
    
    # 查找关键元素
    title_end = None  # 标题结束位置
    table_start = None  # 表格开始
    image_start = None  # 图片开始
    image_end = None  # 图片结束
    stats_start = None  # 统计框开始
    
    for i in range(page_start, page_end):
        line = html_lines[i]
        
        # 标题: "Results | 预测性能对比"
        if title_end is None and '<path fill="#8B0000" d="M69 66L129 66' in line:
            title_end = i + 1  # 标题装饰线之后
            
        # 表格开始: 大框架
        if table_start is None and 'path fill="#F9FAFB" d="M34.5 79.5L924' in line:
            table_start = i
            
        # 图片开始
        if '插入预测性能对比图' in line:
            image_start = i
            
        # 图片结束 (</foreignObject>)
        if image_start and image_end is None and '</foreignObject>' in line:
            image_end = i + 1
            
        # 统计框开始 (第一个统计框的路径)
        if stats_start is None and '<path fill="#8B0000" fill-opacity="0.1" d="M36.75 424.5L315' in line:
            stats_start = i
    
    print(f"  标题结束: line {title_end}")
    print(f"  表格开始: line {table_start}")
    print(f"  图片范围: lines {image_start} - {image_end}")
    print(f"  统计框开始: line {stats_start}")
    
    if not all([title_end, table_start, image_start, image_end, stats_start]):
        print("⚠️  未找到所有关键元素")
        return html_lines
    
    # 新策略:
    # 1. 保留 page_start 到 title_end
    # 2. 跳过 table_start 到 image_start (删除整个表格)
    # 3. 保留图片 (image_start 到 image_end), 但修改 y 坐标
    # 4. 保留统计框 (stats_start 到 page_end)
    
    new_lines = []
    
    for i in range(len(html_lines)):
        # 非第12页内容,直接保留
        if i < page_start or i >= page_end:
            new_lines.append(html_lines[i])
        # 第12页内容
        elif i < title_end:
            # 保留标题
            new_lines.append(html_lines[i])
        elif i >= table_start and i < image_start:
            # 删除表格
            continue
        elif i >= image_start and i < image_end:
            # 保留图片,但修改 y 坐标
            line = html_lines[i]
            # 将 y="320" 改为 y="100"
            line = line.replace('y="320"', 'y="100"')
            new_lines.append(line)
        elif i >= stats_start:
            # 保留统计框
            new_lines.append(html_lines[i])
    
    deleted_count = (image_start - table_start)
    print(f"✅ 第12页:删除了表格 ({deleted_count} 行), 图片上移到 y=100")
    
    return new_lines


def fix_other_pages(html_lines):
    """
    修复第3, 6, 9, 15页的重叠
    策略:直接删除重叠文本元素的行
    """
    
    # 根据检测报告,这些是需要删除的行号
    # 注意:这些行号是基于原始文件的
    
    lines_to_delete = set()
    
    # 第3页: lines 193, 205, 206, 207, 210 (空文本)
    # 但需要删除整个 <g>...</g> 块,所以要扩展范围
    
    # 第6页: line 356 附近
    
    # 第9页: lines 523, 537 附近
    
    # 第15页: 已经在v2中处理过了
    
    # 简化策略:手动标记这些行范围
    page_3_deletes = [193, 205, 206, 207, 210]
    page_6_deletes = [356]
    page_9_deletes = [523, 537]
    
    # 对于每个标记的行,删除它所在的 <g>...</g> 块
    i = 0
    while i < len(html_lines):
        # 检查是否在删除列表中
        if i in page_3_deletes or i in page_6_deletes or i in page_9_deletes:
            # 向前查找 <g transform
            start = i
            while start > 0 and '<g transform=' not in html_lines[start]:
                start -= 1
            
            # 向后查找 </g>
            end = i
            while end < len(html_lines) and '</g>' not in html_lines[end]:
                end += 1
            
            # 标记这个范围内的所有行删除
            for j in range(start, end + 1):
                lines_to_delete.add(j)
        
        i += 1
    
    if lines_to_delete:
        new_lines = [line for i, line in enumerate(html_lines) if i not in lines_to_delete]
        print(f"✅ 其他页面:删除了 {len(lines_to_delete)} 行重叠文本")
        return new_lines
    
    return html_lines


def main():
    input_file = 'CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction.html'
    output_file = 'CausallyExplainableAIonDeepLearningModelforEnergyDemandPrediction_修复版v3.html'
    
    print("=" * 60)
    print("智能修复页面重叠 - 版本3 (激进重构)")
    print("=" * 60)
    
    # 读取HTML
    with open(input_file, 'r', encoding='utf-8') as f:
        html_lines = f.readlines()
    
    print(f"\n原始文件: {len(html_lines)} 行\n")
    
    # 先修复第12页(最严重的)
    html_lines = fix_page_12_aggressive(html_lines)
    
    # 再修复其他页面
    html_lines = fix_other_pages(html_lines)
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(html_lines)
    
    print(f"\n新文件: {len(html_lines)} 行")
    print(f"输出文件: {output_file}")
    print("\n" + "=" * 60)
    print("✅ 修复完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
