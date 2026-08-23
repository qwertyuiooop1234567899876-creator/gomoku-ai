# Native 搜索下沉前基线

## 目标

这份基线用于区分“执行速度不足”和“当前搜索语义无法区分候选”。
固定局面来自 V0.16.8 对 YiXin 第 13 手之前，黑方待走，候选为 F7、
J11。最小夹具保存 12 手有序历史和 Zobrist 哈希，不依赖或恢复
`records/` 中的棋谱。

复现实验：

```powershell
python -B -m tools.native_search_baseline --mode full-window --depths 1-9
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 1000
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 3000
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 6000
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 9000
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 10000
```

## 冷启动全窗口结果

每个候选使用独立 SearchAI 和完整窗口，避免候选之间共享 TT 或 PVS
窄窗。以下是 2026-08-23 在 Python 3.14.7 上的一次基线；墙钟只作本机
参考，节点、分数和 TT 摘要才是等价门禁。表中是未经根 Proof 隔离的
原始 PVS 分数，深度 2/4 的 Mate 带夹值不代表严格证明。

| 深度 | F7 分数 | F7 节点 | J11 分数 | J11 节点 | 相对结果 |
|---:|---:|---:|---:|---:|:---|
| 1 | 3,700 | 1 | -8,200 | 1 | F7 |
| 2 | -999,999,996 | 11 | -999,999,996 | 10 | 同夹值 |
| 3 | 3,500 | 37 | 800 | 46 | F7 |
| 4 | -999,999,994 | 111 | -999,999,994 | 108 | 同夹值 |
| 5 | 11,900 | 595 | -8,000 | 518 | F7 |
| 6 | -892,000 | 1,232 | -990,000 | 1,294 | F7 |
| 7 | 11,000 | 3,533 | -7,100 | 3,381 | F7 |
| 8 | -979,000 | 7,495 | -999,000 | 6,591 | F7 |
| 9 | 3,900 | 22,420 | 100 | 18,036 | F7 |

深度 9 仍未出现 J11 反超。因此当前证据不支持“Native 只要多完成一层
就会自然修正第 13 手”。Native 仍值得做，但初版目标必须先是同语义
等价和深度提升；若更深层仍偏向 F7，需要单独检查评价、选择性候选和
威胁延伸。

## 固定节点结果

双候选迭代 PVS 使用同一棵冷启动搜索树：

| 节点上限 | 完成深度 | 停止原因 | 结果 |
|---:|---:|:---|:---|
| 1,000 | 5 | `node_limit` | F7 11,900；J11 10,900 |
| 3,000 | 6 | `node_limit` | F7 -892,000；J11 -901,100 |
| 6,000 | 7 | `node_limit` | F7 11,000；J11 10,900 |
| 9,000 | 7 | `node_limit` | F7 11,000；J11 10,900 |
| 10,000 | 8 | `requested_depth_completed` | 9,253 节点；F7 -979,000，J11 -987,500 |

节点上限只约束主 PVS；到达上限时保留最后一个完整深度，不把中断层
升级为完成结果。Proof、VCF 和后续复核仍保留各自原有预算语义。

## 热状态限制

仅按棋谱重建棋盘不能重建实战的 TT、history、killer 和 generation。
严格逐手预热在第 3 手已经出现当前引擎 G8 与记录 G7 的分歧，因此工具
主动中止，没有把不同内部状态描述为同一份实战 TT。后续跨实现等价以
冷启动固定节点为第一层；热状态等价需要可序列化状态快照或完全确定的
固定节点整局重放。

## Native 验收顺序

1. 固定深度下保持走法、候选分数、bound、PV、节点语义和完整 TT 摘要一致。
2. 截止继续表示未完成，不能把 UNKNOWN 或中断层转成安全结论。
3. 在不收窄候选与延伸的前提下，同等时间至少多完成一层。
4. 完成更深搜索后重新检查 F7/J11；J11 反超属于棋力验收，不是初版移植的正确性断言。
