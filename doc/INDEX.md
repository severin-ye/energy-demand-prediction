# 📚 文档总览

本文档提供所有项目文档的快速导航和说明。

## 🚀 快速开始（新手必看）

1. **[README.md](../README.md)** - 项目主文档
   - 项目简介
   - 核心特性
   - 30秒快速启动
   - 完整使用流程

2. **[QUICKSTART.md](guides/QUICKSTART.md)** - 详细使用教程
   - 环境安装配置
   - 数据下载和准备
   - 训练模型步骤
   - 推理测试方法
   - 常见问题解答

3. **[QUICK_REFERENCE.md](guides/QUICK_REFERENCE.md)** - 命令速查表
   - 常用命令汇总
   - 参数说明
   - 快捷脚本

## 📖 使用指南

### 入门指南
- **[快速开始](guides/QUICKSTART.md)** - 从零开始的完整教程
- **[快速参考](guides/QUICK_REFERENCE.md)** - 常用命令速查

### 功能指南
- **[HTML可视化演示](guides/HTML_DEMO.md)** - HTML报告使用说明
  - 10步推理流程可视化
  - 查看方式
  - 自定义选项

## 🔬 技术文档

### 论文和理论
- **[论文详细解读](ChatGPT-详细整理论文.md)** - 论文完整解读（教学式）
  - 不用英文术语
  - 不用论文语言
  - 逐步流程讲解
  - 因果推理原理

### 设计和实现
- **[项目设计文档](项目设计文档.md)** - 系统架构设计
  - 总体架构
  - 模块设计
  - 数据流向
  
- **[实现文档](实现文档.md)** - 代码实现说明
  - 核心代码
  - 关键算法
  - 实现细节

### 数据集
- **[UCI数据集说明](数据集说明-UCI_Household.md)** - UCI数据集详解
  - 数据集描述
  - 特征说明
  - 使用方法

## 📊 结果文档

### 训练结果
- **[训练报告](../outputs/training_uci/TRAINING_REPORT.md)** - UCI数据训练详细报告
  - 9步训练流程
  - 性能指标
  - 模型参数
  - 因果网络结果

### 推理结果
- **[推理摘要](../outputs/inference_uci/INFERENCE_SUMMARY.md)** - 推理测试结果摘要
  - 测试概况
  - 性能指标（MAE, RMSE）
  - 状态分布分析
  - DLP特征分析
  - 典型案例
  - 改进建议

- **[HTML可视化指南](../outputs/inference_uci/HTML_VISUALIZATION_GUIDE.md)** - HTML报告详细使用指南

## 📝 进度和总结

- **[实现总结](summaries/IMPLEMENTATION_SUMMARY.md)** - 实现进度汇总
  - 已完成功能
  - 代码统计
  - 关键成果

- **[项目进度](summaries/PROGRESS_SUMMARY.md)** - 整体进度追踪
  - 任务清单
  - 完成情况
  - 待办事项

- **[PROGRESS.md](summaries/PROGRESS.md)** - 简化进度记录

## 🏗️ 项目结构

- **[PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)** - 完整项目结构说明
  - 目录树
  - 文件说明
  - 引用关系
  - 模块依赖图

## 📄 原始论文

- **[能源预测论文.pdf](能源预测--基于深度学习模型的因果可解释人工智能在能源需求预测中的应用.pdf)** - 原始论文PDF

## 🗺️ 文档导航地图

```
README.md (入口)
    ├── 快速开始 → doc/guides/QUICKSTART.md
    ├── 快速参考 → doc/guides/QUICK_REFERENCE.md
    ├── HTML演示 → doc/guides/HTML_DEMO.md
    ├── 项目结构 → PROJECT_STRUCTURE.md
    │
    ├── 技术文档
    │   ├── 论文解读 → doc/ChatGPT-详细整理论文.md
    │   ├── 设计文档 → doc/项目设计文档.md
    │   ├── 实现文档 → doc/实现文档.md
    │   └── 数据集说明 → doc/数据集说明-UCI_Household.md
    │
    └── 结果报告
        ├── 训练报告 → outputs/training_uci/TRAINING_REPORT.md
        └── 推理摘要 → outputs/inference_uci/INFERENCE_SUMMARY.md
```

## 🎯 按需求查找文档

| 我想... | 看这个文档 |
|---------|-----------|
| 快速上手项目 | [README.md](../README.md) |
| 详细安装配置 | [QUICKSTART.md](guides/QUICKSTART.md) |
| 查看常用命令 | [QUICK_REFERENCE.md](guides/QUICK_REFERENCE.md) |
| 理解论文内容 | [论文详细解读](ChatGPT-详细整理论文.md) |
| 了解系统架构 | [项目设计文档](项目设计文档.md) |
| 看代码实现 | [实现文档](实现文档.md) |
| 了解UCI数据集 | [数据集说明](数据集说明-UCI_Household.md) |
| 查看训练结果 | [训练报告](../outputs/training_uci/TRAINING_REPORT.md) |
| 查看推理结果 | [推理摘要](../outputs/inference_uci/INFERENCE_SUMMARY.md) |
| 使用HTML可视化 | [HTML演示](guides/HTML_DEMO.md) |
| 了解项目结构 | [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) |
| 查看实现进度 | [实现总结](summaries/IMPLEMENTATION_SUMMARY.md) |

## 📚 按难度分级

### 🌱 入门级（新手必读）
1. README.md - 项目概览
2. QUICKSTART.md - 快速开始
3. QUICK_REFERENCE.md - 命令速查

### 🌿 进阶级（深入理解）
1. HTML_DEMO.md - 可视化功能
2. 数据集说明-UCI_Household.md - 数据集详解
3. 训练报告 / 推理摘要 - 查看结果

### 🌳 高级级（原理和实现）
1. ChatGPT-详细整理论文.md - 论文解读
2. 项目设计文档.md - 系统设计
3. 实现文档.md - 代码实现
4. PROJECT_STRUCTURE.md - 项目结构

## 🔄 文档更新记录

- **2026-01-16**: 项目文档整理
  - 创建文档总览
  - 整理目录结构
  - 更新所有引用

## 💡 使用建议

1. **第一次使用**: 按顺序阅读 README → QUICKSTART → 实际操作
2. **遇到问题**: 先查 QUICK_REFERENCE，再查具体文档
3. **理解原理**: 阅读论文解读和设计文档
4. **查看结果**: 直接看训练/推理报告
5. **深入学习**: 结合代码和实现文档

---

**最后更新**: 2026-01-16  
**维护者**: 项目团队
