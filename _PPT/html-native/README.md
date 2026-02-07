# 能源预测论文复现 HTML PPT

## 📁 文件结构

```
_PPT/html/
├── presentation.html      # 主PPT文件
├── styles.css            # 样式系统
├── generate_pdf.sh       # PDF生成脚本
└── README.md            # 本文件
```

## 🎨 特点

- ✅ **16:9比例** - 标准PPT尺寸 (13.333in × 7.5in)
- ✅ **38页完整内容** - 覆盖论文所有关键部分
- ✅ **统一风格** - 专业学术配色方案
- ✅ **分页稳定** - 每页固定尺寸，打印友好
- ✅ **响应式布局** - 多栏、表格、卡片、图表等
- ✅ **可直接打印PDF** - 使用Chrome/Chromium headless模式

## 🚀 使用方法

### 方法1: 浏览器直接查看

```bash
cd /home/severin/Codelib/YS/_PPT/html

# 在浏览器中打开
firefox presentation.html
# 或
google-chrome presentation.html
```

**提示**: 可以使用浏览器的"演示模式" (F11全屏) 查看效果

### 方法2: 生成PDF

```bash
cd /home/severin/Codelib/YS/_PPT/html

# 给脚本添加执行权限
chmod +x generate_pdf.sh

# 生成PDF
./generate_pdf.sh
```

**输出**: `能源预测论文复现PPT.pdf`

### 方法3: 手动打印为PDF

1. 在Chrome/Chromium中打开 `presentation.html`
2. 按 `Ctrl+P` (打印)
3. 目标选择 "另存为PDF"
4. 布局选择 "横向"
5. 边距选择 "无"
6. 取消勾选 "页眉和页脚"
7. 点击"保存"

## 📖 PPT内容结构

| 章节 | 页码 | 内容 |
|------|------|------|
| 封面 | 1 | 标题、作者信息 |
| 目录 | 2 | 完整章节导航 |
| **1. Introduction** | 3-6 | 研究背景、动机、贡献 |
| **2. Related Work** | 7-10 | 预测方法演进、XAI对比、创新点 |
| **3. Method** | 11-18 | 系统架构、并行CNN-LSTM、BN构建 |
| **4. Experimental Setup** | 19-22 | 数据集、模型配置、评价指标 |
| **5. Results** | 23-26 | UCI结果、消融实验、XAI稳定性 |
| **6. Analysis** | 27-29 | 成功复现、偏差分析 |
| **7. Issues & Insights** | 30-33 | 核心挑战、研究洞见、未来启发 |
| **8. Conclusion** | 34-36 | 成果总结、未来展望 |
| 参考文献 | 37 | 主要参考文献 |
| 致谢 | 38 | Q&A页 |

## 🎨 样式系统说明

### 颜色主题

- **主题色**: 深蓝 (#1e3a8a) - 标题、强调
- **强调色**: 亮蓝 (#3b82f6) - 次要元素
- **成功**: 绿色 (#10b981) - 正面结果
- **警告**: 橙色 (#f59e0b) - 注意事项
- **危险**: 红色 (#ef4444) - 问题、错误

### 组件库

- **卡片**: `.card`, `.card-primary`, `.card-success`, `.card-warning`
- **信息框**: `.info-box`, `.success-box`, `.danger-box`, `.highlight-box`
- **布局**: `.two-col`, `.three-col`, `.col-6-4`, `.col-4-6`
- **表格**: 自动斑马纹、悬停高亮
- **代码**: 语法高亮、适合公式和算法
- **徽章**: `.badge-primary`, `.badge-success`, `.badge-warning`, `.badge-info`

### 字体大小

```css
--font-title: 48pt;   /* 大标题 */
--font-h1: 36pt;      /* 一级标题 */
--font-h2: 28pt;      /* 二级标题 */
--font-h3: 22pt;      /* 三级标题 */
--font-body: 16pt;    /* 正文 */
--font-small: 14pt;   /* 小字 */
--font-tiny: 12pt;    /* 极小字 */
```

## 🛠️ 自定义修改

### 修改内容

直接编辑 `presentation.html`，每个 `<section class="slide">` 是一页：

```html
<section class="slide">
  <div class="safe">
    <div class="header">
      <div class="section-indicator">章节名</div>
      <div class="page-number">页码</div>
    </div>
    
    <h2 class="slide-title">标题</h2>
    
    <div class="content">
      <!-- 这里放内容 -->
    </div>
    
    <div class="footer">
      <div class="author">左下角文字</div>
      <div class="page-info">右下角页码</div>
    </div>
  </div>
</section>
```

### 修改样式

编辑 `styles.css`：

```css
/* 修改主题色 */
:root {
  --primary: #你的颜色;
  --secondary: #你的颜色;
}

/* 修改字体大小 */
:root {
  --font-body: 18pt;  /* 调大正文 */
}
```

### 添加新页

复制现有的 `<section class="slide">...</section>` 块，修改内容即可。

**注意**: 记得更新页码！

## 📊 导出到PowerPoint

### 方法1: PDF → PPT (推荐)

```bash
# 1. 生成PDF
./generate_pdf.sh

# 2. 使用在线工具转换
# https://www.ilovepdf.com/pdf_to_powerpoint
# 或本地工具: libreoffice

# 3. Linux命令行转换
libreoffice --headless --convert-to pptx 能源预测论文复现PPT.pdf
```

### 方法2: 截图导入

1. 在浏览器中全屏查看每一页 (F11)
2. 使用截图工具 (Flameshot/Shutter) 捕获每页
3. 在PowerPoint中插入图片

### 方法3: 使用转换工具

```bash
# 安装 decktape (需要Node.js)
npm install -g decktape

# 转换为PDF (备用方法)
decktape generic file://$(pwd)/presentation.html output.pdf
```

## 🔧 故障排除

### PDF生成失败

**问题**: `generate_pdf.sh` 报错找不到Chrome

**解决**:
```bash
# Ubuntu/Debian
sudo apt install chromium-browser

# Fedora
sudo dnf install chromium

# Arch Linux
sudo pacman -S chromium
```

### 打印时分页错误

**问题**: PDF页面布局混乱

**解决**: 确保使用Chrome/Chromium打印，设置如下：
- 布局: 横向
- 边距: 无
- 缩放: 100%
- 取消页眉页脚

### 样式不生效

**问题**: 打开HTML后样式丢失

**解决**: 确保 `styles.css` 在同一目录下，路径正确：
```html
<link rel="stylesheet" href="styles.css">
```

## 📝 许可与致谢

本PPT基于论文：
- **Erlangga, G., & Cho, S. B. (2025).** Causally explainable artificial intelligence on deep learning model for energy demand prediction.

样式系统采用现代CSS Grid/Flexbox布局，兼容所有现代浏览器。

---

**更新日期**: 2026年2月5日  
**版本**: v1.0
