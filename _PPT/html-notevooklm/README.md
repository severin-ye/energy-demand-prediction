# NotebookLM PPT HTML 生成项目

## 项目说明

这个文件夹用于根据 NotebookLM 生成的 PPT 截图来创建 HTML 版本的演示文稿。

## 文件说明

- `styles.css` - 完整的样式表文件,包含所有PPT样式定义
- `1-cover-intro.html` - 第1页：封面页
- `2-research-background.html` - 第2页：研究背景（高精度预测与黑盒模型的矛盾）
- 每一页独立成文件，文件名格式：`序号-内容描述.html`

## 工作流程

1. ✅ 样式文件已复制完成
2. ⏳ 等待用户提供每一页的 PPT 截图
3. ⏳ 根据截图内容生成对应的 HTML 代码
4. ⏳ 如有图片需要,先留出占位符,后续插入

## 可用的样式组件

### 页面类型
- `.slide.cover` - 封面页
- `.slide.section-divider` - 章节分隔页
- `.slide` - 普通内容页

### 布局
- `.two-col` - 两栏布局
- `.three-col` - 三栏布局
- `.col-6-4` - 6:4 比例布局
- `.col-4-6` - 4:6 比例布局

### 组件
- `.card` - 卡片
- `.card-primary` - 主题色卡片
- `.card-success` - 成功色卡片
- `.card-warning` - 警告色卡片
- `.highlight-box` - 高亮框
- `.info-box` - 信息框
- `.success-box` - 成功框
- `.danger-box` - 危险框
- `.image-placeholder` - 图片占位符

### 文本样式
- `.slide-title` - 页面标题
- `.slide-subtitle` - 副标题
- `.slide-section` - 小节标题

## 注意事项

- 所有页面尺寸为 16:9 比例 (13.333in × 7.5in)
- 图片占位符会用虚线框表示,方便后续插入实际图片
- 页码和页脚信息需要手动更新

## 下一步

请提供每一页的 PPT 截图,我将根据截图内容生成相应的 HTML 代码。
