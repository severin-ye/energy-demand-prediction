因果驱动的能源需求干预方案生成与自适应推荐系统
基于 Contextual / Slate Bandit 的人机反馈闭环设计（增强反馈版）
1. 研究目标与问题定义

传统的可解释能源需求预测系统主要聚焦于回答
“为什么会发生负荷峰值（Why）”，
但在真实能源管理与调度场景中，更关键的问题是：

“如果我现在改变用电行为，会发生什么变化（What-if）？
以及我应该采取哪一种改变（Which action）？”

因此，本系统将研究目标从因果解释（Causal Explanation）
进一步拓展为：

因果驱动的、可执行的、并可在用户反馈下持续进化的决策推荐系统
（Causal, Actionable, and Adaptive Decision Recommendation）

核心研究问题可形式化为：

在给定上下文条件（季节、时间、建筑规模、当前负荷状态等）下，
如何生成并推荐一组可执行的干预方案，
在峰值削减、能耗降低、用户可接受性与安全约束之间取得最优平衡，
并使系统在真实用户交互中越用越聪明。

2. 系统总体架构

系统由六个核心模块组成，形成一个因果–推荐–反馈闭环：

上下文感知模块（Context Encoder）

干预方案生成模块（Intervention Generator）

因果效果评估模块（Causal Evaluator, BN）

方案排序与推荐模块（Contextual / Slate Bandit）

增强用户反馈采集模块（Enhanced Feedback Collector）

在线学习与生成策略演化模块（Online Update Loop）

整体信息流：

上下文输入
   ↓
方案生成（候选集合）
   ↓
因果干预评估（BN do-calculus）
   ↓
Bandit 排序与 Top-K 推荐
   ↓
用户反馈（选择 / 拒绝原因 / 点赞 / 执行确认）
   ↓
Bandit 更新 + 偏好模型更新 + 生成策略偏置更新

3. 上下文建模（Context Modeling）

上下文向量 
𝑐
c 由三类信息组成：

3.1 环境与时间上下文

季节（Season）

时间段（Hour-of-day）

工作日 / 周末

气候区间（可选）

3.2 建筑与系统特征

建筑类型（住宅 / 工业 / 商业）

建筑规模

是否存在关键负载或刚性负载

3.3 当前负荷与模型内部状态

近期负荷窗口统计特征

能源需求模式（EDP：Peak / Stable / Lower）

深度学习参数离散化特征（DLP）

Attention 类型（Early / Late / Mixed）

CAM 聚类类型（设备使用模式）

最终上下文表示为：

𝑐
=
[
𝑐
env
,
𝑐
building
,
𝑐
load
,
𝑐
DLP
]
c=[c
env
	​

,c
building
	​

,c
load
	​

,c
DLP
	​

]
4. 干预方案生成（Intervention Generation）
4.1 可干预变量定义

基于因果贝叶斯网络结构，变量被划分为：

可干预变量（Actionable）

空调功率档位

高能耗设备启停或延迟

不可干预变量（Contextual）

季节、天气、日期

结果变量（Outcome）

峰值预警概率

总能耗

仅允许对 Actionable 节点执行 do-operator。

4.2 冷启动：人工方案模板库

为保证系统早期的安全性与可控性，采用专家规则模板库进行冷启动。

模板根据上下文分组，例如：

夏季住宅：

T1：空调 Very High → High

T2：空调 High → Medium

工业场景：

T3：生产线 B 启动延迟 30 分钟

T4：高耗能设备错峰启动

模板仅定义干预结构，参数可实例化。

4.3 候选方案生成

在每个决策周期：

根据上下文 
𝑐
c 选择若干模板

进行参数化扩展（不同档位 / 不同延迟）

生成候选集合：

𝑃
(
𝑐
)
=
{
𝜋
1
,
𝜋
2
,
.
.
.
,
𝜋
𝑁
}
P(c)={π
1
	​

,π
2
	​

,...,π
N
	​

}
5. 因果效果评估（Causal Evaluation）
5.1 干预推理

对每个方案 
𝜋
π，通过 BN 执行：

𝑃
(
Peak
∣
𝑑
𝑜
(
𝜋
)
,
𝑐
)
P(Peak∣do(π),c)
𝐸
(
Energy
∣
𝑑
𝑜
(
𝜋
)
,
𝑐
)
E(Energy∣do(π),c)
5.2 因果效果指标

峰值风险降低：

Δ
Peak
=
𝑃
(
Peak
∣
𝑐
)
−
𝑃
(
Peak
∣
𝑑
𝑜
(
𝜋
)
,
𝑐
)
ΔPeak=P(Peak∣c)−P(Peak∣do(π),c)

能耗变化：

Δ
Energy
=
𝐸
(
Energy
∣
𝑐
)
−
𝐸
(
Energy
∣
𝑑
𝑜
(
𝜋
)
,
𝑐
)
ΔEnergy=E(Energy∣c)−E(Energy∣do(π),c)

这些指标构成后续学习与推荐的因果信号。

6. 基于 Slate Bandit 的方案排序与推荐
6.1 推荐建模

在给定上下文 
𝑐
c 下，从候选集合中选择 Top-K 方案列表（Slate）：

𝑆
=
[
𝜋
(
1
)
,
𝜋
(
2
)
,
.
.
.
,
𝜋
(
𝐾
)
]
S=[π
(1)
	​

,π
(2)
	​

,...,π
(K)
	​

]

目标是最大化长期期望收益。

6.2 奖励函数（增强反馈版）

每个方案的即时奖励定义为：

𝑟
(
𝜋
)
=
𝛼
⋅
AcceptIntent
+
𝛽
⋅
Like
−
𝛾
⋅
Dislike
+
𝜆
⋅
Δ
Peak
+
𝜇
⋅
Δ
Energy
−
𝜌
⋅
Cost
−
𝜂
⋅
RejectionPenalty
r(π)=α⋅AcceptIntent+β⋅Like−γ⋅Dislike+λ⋅ΔPeak+μ⋅ΔEnergy−ρ⋅Cost−η⋅RejectionPenalty

其中新增两类关键信号：

7. 增强用户反馈建模（核心扩展）
7.1 反馈类型一：拒绝原因（Preference Decomposition）

当用户对方案选择“不感兴趣 / 拒绝”时，系统提供轻量级原因选项：

太麻烦

不舒服

影响工作 / 生产

不可信

已经在做

学术意义

这相当于将“负反馈”从一个标量，分解为多维偏好标签：

RejectReason
∈
{
Effort
,
Comfort
,
WorkImpact
,
Trust
,
Redundant
}
RejectReason∈{Effort,Comfort,WorkImpact,Trust,Redundant}
系统作用

用于训练 偏好预测模型

𝑃
(
AcceptIntent
∣
𝑐
,
𝜋
)
P(AcceptIntent∣c,π)

用于给生成器提供结构性约束：

“太麻烦” → 限制多变量组合

“不舒服” → 减小调节幅度

“影响工作” → 工业场景下限制某类模板

👉 这一步显著提升了方案生成的可控性与可解释性。

7.2 反馈类型二：执行确认（Intent vs Execution）

系统显式区分两个概念：

Accept-Intent（愿意执行）

Executed（真实执行）

为什么必须区分？

用户可能“愿意”，但未真正执行

若不区分，会导致：

因果效果评估被噪声污染

系统误以为方案“无效”

数据建模

对每个方案记录：

AcceptIntent ∈ {0,1}
Executed ∈ {0,1}


并仅在 Executed = 1 时，使用真实负荷变化更新因果效果评估。

8. Bandit 在线学习与生成策略演化
8.1 Bandit 更新

采用 Contextual Thompson Sampling / LinUCB：

上下文：
𝑐
c

动作：方案 
𝜋
π

奖励：融合因果效果 + 偏好反馈

Bandit 学到的是：

在不同上下文下，哪类方案结构更可能成功。

8.2 生成策略的自适应更新

Bandit 的学习结果反向影响生成阶段：

高收益模板在相似上下文中被更频繁采样

被多次标记为“太麻烦 / 不舒服”的模板被降权

系统逐步收敛到高因果收益 + 高可接受性的方案空间

9. 探索–利用平衡与安全性

采用 Thompson Sampling 保持不确定性探索

设置最小探索比例（如 20%）

所有方案必须通过 Action Masking：

不违反安全、舒适、生产约束

不允许干预不可控变量

10. 实验与评估指标（增强版）
10.1 决策效果

平均峰值风险降低率

总能耗节省比例

反弹峰值发生率

10.2 用户层面

Accept-Intent 率

Executed 率

各拒绝原因分布随时间变化

10.3 学习效率

冷启动 → 稳定阶段的收敛速度

方案生成空间的熵下降趋势（是否更“聪明”）

11. 总结

本系统将因果可解释能源预测模型，进一步演化为：

一个能够生成方案、评估方案、学习人类偏好，并在真实交互中持续进化的因果决策系统。

其关键创新在于：

使用因果推理而非相关性评估方案效果

通过 Contextual / Slate Bandit 实现在线学习

引入拒绝原因与执行确认，显著降低反馈噪声

使方案生成本身在反馈中持续变聪明