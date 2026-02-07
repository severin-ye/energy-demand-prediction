# 手动生成PDF指南

由于系统上没有安装PDF生成工具，这里提供详细的手动操作步骤。

## 🎯 最简单方法：浏览器打印

### 步骤1: 在浏览器中打开HTML

```bash
cd /home/severin/Codelib/YS/_PPT/html

# 方法A: 如果有Firefox
firefox presentation.html &

# 方法B: 如果有其他浏览器
xdg-open presentation.html
```

如果上述命令不工作，手动操作：
1. 打开任意浏览器
2. 按 `Ctrl+L` 进入地址栏
3. 输入: `file:///home/severin/Codelib/YS/_PPT/html/presentation.html`
4. 回车打开

### 步骤2: 打印为PDF

在浏览器中：

1. **按 `Ctrl+P`** 打开打印对话框

2. **设置打印选项**:
   - **目标/打印机**: 选择 "另存为PDF" 或 "Print to PDF"
   - **布局**: 选择 "横向 (Landscape)"
   - **页面**: 全部
   - **边距**: 选择 "无" 或 "最小"
   - **比例**: 100%
   - **背景图形**: 勾选（确保颜色正常）
   - **页眉和页脚**: 取消勾选

3. **点击"保存"或"打印"**

4. **选择保存位置和文件名**:
   ```
   文件名: 能源预测论文复现PPT.pdf
   位置: /home/severin/Codelib/YS/_PPT/html/
   ```

5. **确认保存**

### 步骤3: 验证结果

```bash
cd /home/severin/Codelib/YS/_PPT/html
ls -lh *.pdf
```

应该看到生成的PDF文件。

---

## 🔧 安装PDF生成工具（推荐）

如果你经常需要生成PDF，建议安装专用工具：

### 方案1: 安装 Chromium (最推荐)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install chromium-browser

# 安装后运行脚本
./generate_pdf.sh
```

**优点**: 
- ✅ 生成速度快
- ✅ 效果最好
- ✅ 支持复杂CSS
- ✅ 脚本自动化

### 方案2: 安装 wkhtmltopdf

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install wkhtmltopdf

# 安装后运行脚本
./generate_pdf.sh
```

**优点**:
- ✅ 专门为HTML→PDF优化
- ✅ 命令行友好
- ✅ 无需浏览器

### 方案3: 安装 WeasyPrint (Python)

```bash
# 使用pip安装
pip3 install weasyprint

# 或系统包管理器
sudo apt install python3-weasyprint

# 安装后运行脚本
./generate_pdf.sh
```

**优点**:
- ✅ Python生态
- ✅ 易于集成
- ✅ 轻量级

---

## 📋 快速命令对照表

### 检查已安装的工具

```bash
# 检查所有可能的PDF工具
which chromium chromium-browser google-chrome firefox wkhtmltopdf
python3 -c "import weasyprint; print('WeasyPrint已安装')" 2>/dev/null
```

### 直接使用浏览器命令行（如果已安装）

```bash
# 使用Firefox (如果有)
firefox --print-to-pdf=/tmp/output.pdf presentation.html

# 使用Chromium (安装后)
chromium --headless --print-to-pdf=output.pdf presentation.html
```

---

## 🎨 打印设置详解

### 关键设置项

| 设置项 | 推荐值 | 说明 |
|--------|--------|------|
| 页面方向 | **横向 (Landscape)** | 16:9比例必须横向 |
| 纸张大小 | 自定义 13.333 × 7.5 英寸 | 标准PPT尺寸 |
| 边距 | **无 (None)** | 避免内容被裁剪 |
| 缩放 | **100%** | 保持原始尺寸 |
| 背景图形 | ✅ **必须勾选** | 否则没有颜色 |
| 页眉页脚 | ❌ **必须取消** | 避免额外文字 |

### 常见问题修复

**问题1: PDF颜色全是黑白**
- **解决**: 勾选"背景图形"或"打印背景色"

**问题2: 内容被裁剪**
- **解决**: 设置边距为"无"或"0"

**问题3: 分页错乱**
- **解决**: 确保缩放为100%

**问题4: 字体模糊**
- **解决**: 在高级设置中选择"高质量打印"

---

## 💻 Linux系统特定命令

### Ubuntu/Debian

```bash
# 推荐：安装Chromium
sudo apt update && sudo apt install -y chromium-browser

# 或安装wkhtmltopdf
sudo apt install -y wkhtmltopdf
```

### Fedora

```bash
# 安装Chromium
sudo dnf install -y chromium

# 或安装wkhtmltopdf
sudo dnf install -y wkhtmltopdf
```

### Arch Linux

```bash
# 安装Chromium
sudo pacman -S chromium

# 或安装wkhtmltopdf
sudo pacman -S wkhtmltopdf
```

---

## 🚀 完整自动化示例

安装Chromium后，一键生成：

```bash
# 1. 安装 (只需一次)
sudo apt install chromium-browser

# 2. 进入目录
cd /home/severin/Codelib/YS/_PPT/html

# 3. 生成PDF
./generate_pdf.sh

# 4. 查看结果
evince 能源预测论文复现PPT.pdf &
```

---

## 📝 总结

**立即可用** (无需安装):
- ✅ 浏览器手动打印 (Ctrl+P)

**安装后自动化** (推荐):
- ✅ `sudo apt install chromium-browser`
- ✅ 运行 `./generate_pdf.sh`

**当前状态**:
- ⏳ 等待安装PDF工具
- 或立即使用手动方法

选择最适合你的方式！🎯
