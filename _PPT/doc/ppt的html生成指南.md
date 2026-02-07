核心思想

用浏览器的打印引擎把 HTML 渲染成 分页稳定的 PDF（每个 slide = PDF 一页）。

再把 PDF 按页导入 PPT（常见做法：每页转成高分辨率图片后铺满 slide，或直接插入 PDF 页）。

这条路线的优势是“视觉一致性极强”（因为你把版式锁死在 PDF 渲染结果里），缺点是 PPT 内可编辑性弱（通常就是一整页图/矢量块）。

Part 1：HTML → PDF（最关键：分页与尺寸）
1) 固定“画布尺寸”＝PPT 16:9

PPT 16:9 常用画布（英寸）：

13.333in × 7.5in

你要让 PDF 的每页就是这个尺寸（不是 A4！）。

✅ 推荐：用 @page 固定 PDF 页面尺寸

在你的打印 CSS 里写：

/* print.css */
@page {
  size: 13.333in 7.5in;  /* 16:9 PPT */
  margin: 0;             /* 强制无页边距，边距由内容容器控制 */
}

@media print {
  html, body {
    width: 13.333in;
    height: 7.5in;
    margin: 0;
    padding: 0;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  .slide {
    width: 13.333in;
    height: 7.5in;
    page-break-after: always; /* 每个 slide 强制分页 */
    position: relative;
    overflow: hidden;         /* 禁止溢出影响后续页 */
  }

  .slide:last-child { page-break-after: auto; }
}

2) 统一版式：用“安全区容器”控制边距

不要依赖浏览器默认 margin。建议每页都用同一安全区：

:root{
  --safe-x: 0.6in;  /* 左右安全边距 */
  --safe-y: 0.45in; /* 上下安全边距 */
  --font: "Inter", "Noto Sans", "Arial", sans-serif;
}

.slide .safe {
  position: absolute;
  left: var(--safe-x);
  right: var(--safe-x);
  top: var(--safe-y);
  bottom: var(--safe-y);
  font-family: var(--font);
}


对应 HTML：

<section class="slide">
  <div class="safe">
    <h1 class="title">标题</h1>
    <div class="content two-col">
      <div>左栏</div>
      <div>右栏</div>
    </div>
    <div class="footer">页脚 / 页码</div>
  </div>
</section>


这样你就能保证：每一页的内容对齐、边距、布局一致。

3) 分页控制：确保“1 页 = 1 slide”

必须做到：

每个 .slide 固定高度（7.5in）

强制分页（page-break-after: always）

不要让一个 slide 自己跨页。跨页会让后续 PPT 页顺序错乱、风格不一致。

溢出策略（建议你必须做）

列表太长：拆成两页（两个 slide）

表格太大：改两栏/缩列/拆表

图片太高：裁剪（cover）或缩放（contain）

可以在 CSS 里强制溢出隐藏（上面已经加了 overflow:hidden），再用自动化检测来报警（后面给你方案）。

4) 字体一致：避免“换机变形”

PDF 一致性最大的坑就是字体替换导致换行变化。

最稳做法（推荐）

把字体文件放进项目，用 @font-face 引入（并确保 headless Chrome 能访问到字体文件）。

或在 Docker/服务器里安装字体，固定导出环境。

示例：

@font-face{
  font-family: "Inter";
  src: url("./fonts/Inter-Regular.woff2") format("woff2");
  font-weight: 400;
  font-style: normal;
}
@font-face{
  font-family: "Inter";
  src: url("./fonts/Inter-SemiBold.woff2") format("woff2");
  font-weight: 600;
  font-style: normal;
}