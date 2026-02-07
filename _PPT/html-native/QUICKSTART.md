# 🎯 快速开始指南

## ✅ 已完成的工作

我已经为你生成了一个完整的 HTML PPT 系统，包含：

1. **38页完整PPT内容** - 涵盖论文所有核心部分
2. **专业样式系统** - 统一的16:9格式，学术风格配色
3. **PDF生成脚本** - 一键生成可打印的PDF文件
4. **完整文档** - README使用说明

## 📂 文件位置

```
/home/severin/Codelib/YS/_PPT/html/
├── presentation.html      # 主PPT文件 (38页)
├── styles.css            # 样式系统
├── generate_pdf.sh       # PDF生成脚本
└── README.md            # 详细使用说明
```

## 🚀 立即查看

### 方法1: 浏览器预览

```bash
cd /home/severin/Codelib/YS/_PPT/html

# Firefox
firefox presentation.html

# Chrome
google-chrome presentation.html
```

按 `F11` 进入全屏模式，效果更佳！

### 方法2: 生成PDF (推荐)

```bash
cd /home/severin/Codelib/YS/_PPT/html

# 生成PDF
./generate_pdf.sh

# 查看PDF
evince 能源预测论文复现PPT.pdf
```

## 📋 PPT内容概览

| 部分 | 页数 | 核心内容 |
|------|------|----------|
| **封面** | 1 | 标题、论文信息 |
| **目录** | 1 | 章节导航 |
| **Introduction** | 4页 | 背景、动机、贡献 |
| **Related Work** | 4页 | 方法演进、XAI对比 |
| **Method** | 7页 | 并行架构、BN、因果推断 |
| **Experimental Setup** | 4页 | 数据集、配置、指标 |
| **Results** | 4页 | 实验结果、消融、稳定性 |
| **Analysis** | 3页 | 成功与偏差分析 |
| **Issues & Insights** | 4页 | 挑战、洞见、启发 |
| **Conclusion** | 3页 | 总结、展望、参考文献 |
| **致谢** | 1 | Q&A页 |

**总计**: 38页

## 🎨 样式特点

- ✅ **16:9标准比例** (13.333in × 7.5in)
- ✅ **专业配色** 深蓝主题 + 多彩强调
- ✅ **丰富组件** 表格、卡片、代码块、公式
- ✅ **章节分隔** 清晰的章节过渡页
- ✅ **页眉页脚** 一致的导航和页码
- ✅ **打印友好** 固定分页，无溢出

## 🔄 下一步

### 如果需要修改内容

编辑 [presentation.html](presentation.html)，找到对应的 `<section class="slide">` 修改即可。

### 如果需要调整样式

编辑 [styles.css](styles.css)，修改颜色、字体等。

### 如果需要导出到PowerPoint

```bash
# 方法1: 先生成PDF
./generate_pdf.sh

# 方法2: 使用在线工具
# 访问 https://www.ilovepdf.com/pdf_to_powerpoint
# 上传生成的PDF，转换为PPTX

# 方法3: LibreOffice转换
libreoffice --headless --convert-to pptx 能源预测论文复现PPT.pdf
```

## 💡 使用建议

### 演示时

1. 使用浏览器全屏模式 (F11)
2. 可以用 `PageDown`/`PageUp` 或鼠标滚轮翻页
3. 效果和PowerPoint几乎一样

### 打印时

1. 使用 `generate_pdf.sh` 脚本生成PDF
2. 或在Chrome中 `Ctrl+P` → "另存为PDF"
3. 确保设置：横向、无边距、无页眉页脚

### 分享时

- **给导师**: 发送生成的PDF文件
- **给同学**: 分享HTML文件(包含styles.css)
- **在线展示**: 可以部署到GitHub Pages

## 📞 如需帮助

详细使用说明请查看 [README.md](README.md)

---

**生成时间**: 2026年2月5日  
**状态**: ✅ 完成  
**质量**: 38页完整内容，格式统一，可直接使用
