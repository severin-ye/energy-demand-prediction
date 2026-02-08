#!/usr/bin/env node

/**
 * PPT页面溢出检测工具
 * 使用Puppeteer检测每个页面的实际渲染高度
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const MAX_CONTENT_HEIGHT = 6.8 * 96; // 6.8in * 96dpi = 652.8px
const SLIDE_WIDTH = 13.333 * 96; // 13.333in * 96dpi
const SLIDE_HEIGHT = 8.333 * 96; // 8.333in * 96dpi

const PAGES = Array.from({ length: 20 }, (_, i) => {
  const files = [
    '1-cover-intro.html',
    '2-research-background.html',
    '3-core-contributions.html',
    '4-method-overview.html',
    '5-prediction-model-architecture.html',
    '6-feature-discretization.html',
    '7-structure-learning-bn.html',
    '8-dlp-semantic-interpretation.html',
    '9-experimental-setup.html',
    '10-results-uci-prediction.html',
    '11-results-consistency-analysis.html',
    '12-case-study-peak-warning.html',
    '13-ablation-study.html',
    '14-replication-summary.html',
    '15-core-problem-performance.html',
    '16-implementation-details.html',
    '17-solutions-attempted.html',
    '18-future-work-overview.html',
    '19-system-architecture.html',
    '20-technical-innovations.html'
  ];
  return files[i];
});

async function checkPage(browser, filename) {
  const page = await browser.newPage();
  await page.setViewport({
    width: Math.round(SLIDE_WIDTH),
    height: Math.round(SLIDE_HEIGHT),
    deviceScaleFactor: 1,
  });

  const filePath = path.join(__dirname, filename);
  await page.goto(`file://${filePath}`, { waitUntil: 'networkidle0' });

  const metrics = await page.evaluate((maxHeight) => {
    const content = document.querySelector('.content');
    const safe = document.querySelector('.safe');
    
    if (!content) {
      return { error: '找不到 .content 元素' };
    }

    const contentHeight = content.scrollHeight;
    const safeHeight = safe ? safe.scrollHeight : 0;
    const overflow = contentHeight - maxHeight;

    // 获取所有子元素信息
    const children = Array.from(content.children).map(child => ({
      tag: child.tagName,
      height: child.scrollHeight,
      class: child.className
    }));

    return {
      contentHeight,
      safeHeight,
      overflow,
      percentage: ((contentHeight / maxHeight) * 100).toFixed(1),
      children
    };
  }, MAX_CONTENT_HEIGHT);

  await page.close();

  return {
    filename,
    ...metrics,
    status: metrics.error ? 'error' : (
      metrics.overflow > 0 ? 'overflow' : 
      metrics.overflow > -50 ? 'warning' : 'ok'
    )
  };
}

async function main() {
  console.log('🚀 开始检测PPT页面溢出情况...\n');
  console.log(`📏 最大内容高度限制: ${MAX_CONTENT_HEIGHT.toFixed(1)}px (6.8in)\n`);

  const browser = await puppeteer.launch({ headless: 'new' });

  const results = [];
  for (const filename of PAGES) {
    const result = await checkPage(browser, filename);
    results.push(result);

    const statusIcon = result.status === 'ok' ? '✅' : 
                       result.status === 'warning' ? '⚠️' : '❌';
    
    if (result.error) {
      console.log(`${statusIcon} ${filename}`);
      console.log(`   ERROR: ${result.error}\n`);
    } else {
      console.log(`${statusIcon} ${filename}`);
      console.log(`   高度: ${result.contentHeight.toFixed(1)}px / ${MAX_CONTENT_HEIGHT.toFixed(1)}px (${result.percentage}%)`);
      if (result.overflow > 0) {
        console.log(`   ⚠️  超出: ${result.overflow.toFixed(1)}px`);
      } else {
        console.log(`   ✓  剩余: ${Math.abs(result.overflow).toFixed(1)}px`);
      }
      console.log();
    }
  }

  await browser.close();

  // 生成摘要
  console.log('\n' + '='.repeat(60));
  console.log('📋 检测汇总\n');

  const overflowPages = results.filter(r => r.status === 'overflow');
  const warningPages = results.filter(r => r.status === 'warning');
  const okPages = results.filter(r => r.status === 'ok');

  console.log(`❌ 溢出页面: ${overflowPages.length} 个`);
  console.log(`⚠️  警告页面: ${warningPages.length} 个`);
  console.log(`✅ 正常页面: ${okPages.length} 个\n`);

  if (overflowPages.length > 0) {
    console.log('需要修复的页面:');
    overflowPages.forEach(p => {
      console.log(`  • ${p.filename} (超出 ${p.overflow.toFixed(1)}px)`);
    });
  }

  // 保存详细报告
  fs.writeFileSync(
    path.join(__dirname, 'overflow_report.json'),
    JSON.stringify(results, null, 2)
  );
  console.log('\n📄 详细报告已保存到: overflow_report.json');
}

main().catch(console.error);
