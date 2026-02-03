#!/bin/bash
# 项目清理脚本 - 删除过时的代码和文档

set -e

PROJECT_ROOT="/home/severin/Codelib/YS"
cd "$PROJECT_ROOT"

echo "================================"
echo "开始清理项目过时文件"
echo "================================"

# 创建备份目录
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo ""
echo "[1/6] 清理doc/目录下的过程性文档..."
# 保留核心文档，移除过程记录
DOCS_TO_REMOVE=(
    "doc/实验结果分析报告.md"
    "doc/归一化修正实验记录.md"
    "doc/归一化流程检查.md"
    "doc/改造验证报告.md"
    "doc/论文复刻差异分析.md"
    "doc/论文复刻改造总结.md"
    "doc/论文表3表4分析与复现方案.md"
    "doc/达到论文性能行动计划.md"
    "doc/可视化对比.md"
    "doc/可视化改进说明.md"
    "doc/HTML可视化完整版说明.md"
    "doc/ChatGPT-详细整理论文.md"
)

for file in "${DOCS_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        echo "  移除: $file"
        mv "$file" "$BACKUP_DIR/"
    fi
done

echo ""
echo "[2/6] 清理outputs/下的旧日志文件..."
OLD_LOGS=(
    "outputs/ablation_final.log"
    "outputs/ablation_flatten_fix.log"
    "outputs/ablation_minmax.log"
    "outputs/ablation_seed42.log"
    "outputs/ablation_study_fixed.log"
    "outputs/ablation_study_full.log"
    "outputs/ablation_val_mae.log"
    "outputs/parallel_only.log"
)

for log in "${OLD_LOGS[@]}"; do
    if [ -f "$log" ]; then
        echo "  移除: $log"
        rm -f "$log"
    fi
done

echo ""
echo "[3/6] 清理outputs/ablation/下的旧模型..."
# 保留ABLATION_REPORT.md和ablation_results.csv，移除旧模型
if [ -d "outputs/ablation/parallel-att" ]; then
    echo "  移除: outputs/ablation/parallel-att/"
    mv outputs/ablation/parallel-att "$BACKUP_DIR/"
fi
if [ -d "outputs/ablation/serial" ]; then
    echo "  移除: outputs/ablation/serial/"
    mv outputs/ablation/serial "$BACKUP_DIR/"
fi
if [ -d "outputs/ablation/serial-att" ]; then
    echo "  移除: outputs/ablation/serial-att/"
    mv outputs/ablation/serial-att "$BACKUP_DIR/"
fi

echo ""
echo "[4/6] 清理scripts/下的冗余测试脚本..."
SCRIPTS_TO_REMOVE=(
    "scripts/test_parallel_only.py"
    "scripts/test_parallel_eval.py"
    "scripts/train_parallel_only.py"
    "scripts/test_modifications.py"
    "scripts/quick_validation.py"
    "scripts/test_output_structure.py"
    "scripts/run_ablation_study.py"
)

for script in "${SCRIPTS_TO_REMOVE[@]}"; do
    if [ -f "$script" ]; then
        echo "  移除: $script"
        mv "$script" "$BACKUP_DIR/"
    fi
done

echo ""
echo "[5/6] 清理logs/目录..."
if [ -d "logs" ]; then
    echo "  移除整个logs/目录（旧日志）"
    mv logs "$BACKUP_DIR/"
fi

echo ""
echo "[6/6] 清理其他临时文件..."
TEMP_FILES=(
    "tree.md"
    "src/visualization/inference_visualizer_old.py"
)

for file in "${TEMP_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  移除: $file"
        mv "$file" "$BACKUP_DIR/"
    fi
done

echo ""
echo "================================"
echo "清理完成！"
echo "================================"
echo ""
echo "备份目录: $BACKUP_DIR"
echo "如果需要恢复，可以从备份目录中找回文件"
echo ""
echo "清理摘要："
echo "- doc/: 移除11个过程性文档"
echo "- outputs/: 移除8个旧日志文件"
echo "- outputs/ablation/: 移除3个旧模型目录"
echo "- scripts/: 移除7个冗余测试脚本"
echo "- logs/: 移除整个目录"
echo "- 其他: 移除2个临时文件"
echo ""
echo "保留的核心文件："
echo "- doc/gpt-原文翻译.md (论文翻译)"
echo "- doc/项目设计文档.md"
echo "- doc/数据集说明-UCI_Household.md"
echo "- doc/guides/ (快速入门指南)"
echo "- src/ (所有源代码)"
echo "- scripts/train_*.py (新的训练脚本)"
echo "- scripts/run_ablation_comparison.py (新的消融实验脚本)"
echo ""
