# 📡 Andy 的 AI 早报

每天自动采集 AI 领域资讯，由 DeepSeek 筛选、总结、给出应用价值与商业机会，同步生成英文版和 10-15 分钟的语音播客。

聚焦赛道：AI 视频生成 · 图像生成 · 模型 API 与定价 · AI Agent · AI 创作工具 · 多模态模型

- 🌐 网站：https://alicejacklead-bit.github.io/ai-daily/
- 🎧 播客 RSS：https://alicejacklead-bit.github.io/ai-daily/podcast.xml

## 每期内容

每条资讯包含：一句话事实、重要性评分、核心价值、应用价值、可交叉核实的来源链接。
每期末尾：今日观察、商业机会、研究建议。

## 质量控制

- 时间窗：新闻类严格 24 小时；官方博客与周更 Newsletter 72 小时；无发布时间的条目直接丢弃
- 去重：当天按 URL/标题去重；跨天与最近 5 期比对，已报道过的链接直接剔除，同一事件除非有实质新进展否则不重复
- 来源核实：模型输出的每个链接必须出自当天抓取的原始数据，编造的链接自动删除，无可信来源的条目整条删除

## 运行方式

| 工作流 | 作用 |
|---|---|
| `Test run (no publish)` | 手动测试，结果打包下载，不提交、不发布 |
| `AI Daily Digest` | 正式生成并提交，触发 GitHub Pages 发布 |
| `Keepalive` | 防止仓库因长期无活动被停用定时任务 |

需要的 Secret：`API_KEY`（DeepSeek）。

---

Fork 自 [Jimmuji/ai-daily-digest](https://github.com/Jimmuji/ai-daily-digest)，MIT License。
