# TechCompass — 数据源策略（v1.1 重新定位）

## 1. 定位调整

之前项目数据主干是「手写样本 + 历史评测集」，缺少“实时、可公开”支撑。
v1.1 重新明确：

> TechCompass = 基于可公开、可实时获取的多源数据，做数码消费品舆情侦察
> 与行动建议的 Agent。

## 2. 三层数据源

### 第一层：实时公开新闻（主数据源）

- **GNews（主）**：公开 API，按关键词实时搜新闻。已验证稳定，返回最新（如 2026-10-09）科技/数码新闻。
- **NewsAPI（候选）**：公开 API，功能更强；但当前网络环境访问 newsapi.org 超时，暂不接入主链路，待网络允许再启用。

### 第二层：历史评论库（评测/标签基准）

- 用于构建评测集与分类/召回基准，不做“今日舆情”主干。
- 候选：Amazon Reviews 2023 官方公开数据集（覆盖到 2024）。
- 现状：官方数据集因旧脚本格式被 `datasets 5.1.0` 弃用，接入待处理。

### 第三层：用户讨论/公告（后续候选）

- Reddit 公开 API、RSS 公开订阅。
- 后续按产品关键词（earbuds / headphones / smartwatch / charger）接入。

## 3. 已实现的代码

| 文件 | 作用 |
| --- | --- |
| `src/adapters/realtime_base.py` | 统一 RealtimeSource 接口 + RealtimeItem |
| `src/adapters/realtime_gnews.py` | GNews 实时新闻源 |
| `src/adapters/realtime_newsapi.py` | NewsAPI 实时新闻源（已实现，网络待验证） |
| `src/tools/domain_tools.py` | `search_news` 已切到 GNewsSource |

## 4. 决策原则

- 只使用官方公开 API / 官方公开数据集，不做网页爬取（避免反爬、合规、不可复现问题）。
- 一个源失败自动降级，不影响系统稳定性。
- 实时源负责“当下”，历史库负责“评测与基准”。

## 5. 下一步

- [ ] NewsAPI 网络可达后启用为第二实时源。
- [ ] 用官方评论数据集重建评测集（替换手写）。
- [ ] 接入 Reddit/RSS 作为用户讨论补充。

## 6. Reddit 接入说明（新增）

- 已安装 `praw==8.0.3`。
- 已实现 `src/adapters/realtime_reddit.py`（RealtimeSource 接口）。
- 仍需用户创建 Reddit script app，填 `REDDIT_CLIENT_ID/SECRET/USER_AGENT`。
- 配置缺失时 `search()` 安全返回空列表。
- 接入后可按关键词搜索 subreddit="all"，获得帖子 title/selftext/permalink/score/num_comments。
