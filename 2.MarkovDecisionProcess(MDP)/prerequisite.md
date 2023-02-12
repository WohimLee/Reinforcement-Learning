&emsp;
# Prerequisite
# Agent-Environment

>马尔科夫决策过程（MDP）
- Agent: 智能体
- Environment: 环境
<div align=center>
    <image src="./imgs/agent-environment.png" width=500>
</div>

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, ...$$

- $t$: $t=0, 1, 2, 3, ...$ Agent 和 Environment 发生交互的 `离散时刻`

- $S_t \in \mathcal{S}$: Agent 体观察到所在的 `Environment 状态的某种特征表达`
- $A_t \in \mathcal{A}(s)$: 在 $S_t$ 下选择的一个动作
- $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$: 下一时刻，作为 $A_t$ 的结果，Agent 接收到一个数值化的收益（Reward），并进入到一个新的状态 $S_{t+1}$

&emsp;
>有限马尔科夫
- 在有限 MDP 中，State、Action 和 Reward的集合（$\mathcal{S}$、$\mathcal{A}$、$\mathcal{R}$）都只有有限个元素
- 随机变量 $R_t$ 和 $S_t$ 具有定义明确的离散概率分布
- 随机变量 $R_t$ 和 $S_t$ 只以来与前继状态和动作
- 综上，给定前继状态和动作的值时，这些随机变量的特定值，$s' \in \mathcal{S}$ 和 $r \in \mathcal{R}$ 在 t 时刻出现的概率是：
$$p(s', r | s, a) \doteq P\{S_t = s', R_t=r | S_{t-1}=s, A_{t-1}a\}$$


&emsp;
>马尔科夫性
- State 必须包括过去 Agent 和 Environment 交互的方方面面的信息，这些信息会对未来产生一定影响
- 这样的状态就被认为具有 `马尔科夫性`

&emsp;
# Goals and Rewards
>Goals 目标
- 最大化 Agent 接收到的 Reward（标量信号）累积和的 `概率期望值`

&emsp;
# Returns