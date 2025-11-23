# STARK Prompts 设计分析

`prompts/stark.py` 文件定义了 **STARK**（一个多智能体 GPU 内核优化系统）中三个核心智能体（Agent）的提示词（Prompts）。

这套设计采用了 **角色分离（Separation of Concerns）** 的策略，将复杂的优化任务拆解为“规划”、“编码”和“调试”三个明确的阶段。

以下是详细的设计解读：

### 1. Plan Agent (规划智能体)
这个智能体是大脑，负责制定战略。

*   **角色定位 (`PLAN_SYSTEM_PROMPT`)**:
    *   **身份**: STARK 中的 Plan Agent。
    *   **目标**: 提出**单一且影响力最大**的优化方案，并使用“锚点（Anchors）”明确标记代码中需要修改的**确切位置**。
    *   **风格**: 简洁、外科手术般精准、关注指标。
    *   **温度建议**: 0.8（在此阶段允许一定的创造性以探索新的优化路径）。
*   **任务输入 (`PLAN_PROMPT_TMPL`)**:
    *   **Current kernel**: 当前待优化的代码节点。
    *   **Local history**: 局部历史（子节点/兄弟节点），用于避免重复错误。
    *   **Global leaders**: 全局领先者（排行榜上的 Top 候选），用于借鉴成功经验。
*   **输出要求**:
    1.  **JSON 计划**: 包含 `targets`（改什么）、`tactics`（怎么改）、`risks`（风险）、`expected_gain`（预期收益）。这让下游知道**为什么**要改。
    2.  **带锚点的代码**: 使用 `<<<IMPROVE BEGIN:id>>> ... <<<IMPROVE END:id>>>` 标记出需要修改的代码块。未标记区域保持原样。

### 2. Code Agent (编码智能体)
这个智能体是执行者，负责战术落地。

*   **角色定位 (`CODE_SYSTEM_PROMPT`)**:
    *   **身份**: STARK 中的 Code Agent。
    *   **目标**: **确定性地**实现 Plan Agent 的计划。严格限制在锚点区域内修改，保持外部 API 和 Tensor 契约不变。
    *   **风格**: 低温度（≈0.1），严禁在锚点外进行推测性编辑。
*   **任务输入 (`CODE_PROMPT_TMPL`)**:
    *   **Plan JSON**: 来自 Plan Agent 的具体指导。
    *   **Anchored scaffold**: 带有锚点标记的代码骨架。
    *   **Neighbour kernels**: 相关的邻居内核代码，仅作为具体实现的参考（如线程块大小、内存分级写法）。
*   **输出要求**:
    *   返回完整的、可运行的 Python 代码，且必须**移除所有锚点标记**。

### 3. Debug Agent (调试智能体)
这个智能体是维修工，负责修复错误。

*   **角色定位 (`DEBUG_SYSTEM_PROMPT`)**:
    *   **触发时机**: 当一个有希望的子节点运行失败时介入。
    *   **目标**: 修复编译或运行时错误，同时**保留**原计划的优化意图。
    *   **原则**: 优先保证正确性，避免引入新特性。
*   **任务输入 (`DEBUG_PROMPT_TMPL`)**:
    *   **Error log**: 错误日志截断。
    *   **Current broken kernel**: 当前损坏的代码。
    *   **Context**: 兄弟或领先节点的代码，作为修复 CUDA 细节的参考线索。
    *   **Plan hint**: 原始计划的提示，确保修复不偏离初衷。
*   **输出要求**:
    *   仅返回修复后的代码。

### 设计亮点总结

1.  **上下文感知 (Context-Awareness)**:
    *   Prompts 显式地注入了 `Local history` 和 `Global leaders`。这意味着 Agent 不是在真空中工作，而是能从过去的成功（Leaders）和失败（History）中学习。

2.  **锚点机制 (Anchoring Mechanism)**:
    *   这是设计的核心。Plan Agent 不直接写完代码，而是圈定“手术范围”（Anchors）。Code Agent 只能在圈定范围内操作。这极大地减少了 LLM 幻觉导致破坏无关代码（如数据加载、验证逻辑）的风险。

3.  **两阶段生成 (Plan -> Code)**:
    *   将“思考做什么”（高温度）和“具体怎么写”（低温度）分开。Plan Agent 可以天马行空地构思优化策略，而 Code Agent 则像一个严谨的工程师将其落地。

4.  **结构化通信**:
    *   Agent 之间通过 JSON 和特定的标记格式（Anchors）进行通信，而不是自然语言对话，这提高了自动化系统的鲁棒性。

----