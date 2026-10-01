# ProjectB ToolRL rlla_4k SFT 数据审计

本报告只审计远程 parquet，不生成 SFT 数据文件，不修改 parquet、reward manager、verl 源码或训练环境。

审计文件：

    /root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet

审计结果：3920 行，1 个 row group。所有结论来自远程 modern 环境中的只读读取。

## 1. 数据字段

| 字段 | 类型 | 实际内容 |
|---|---|---|
| data_source | string | 全部为 rlla |
| prompt | list of {role, content} | 每行恰好两个消息，顺序为 system、user |
| ability | string | 全部为 math |
| reward_model | struct | ground_truth 和 style |
| extra_info | struct | index、input、instruction、output、split |

### response 字段的实际位置

parquet 没有顶层 response 列。可直接作为 chosen response 的字段是：

1. reward_model.ground_truth
2. extra_info.output

两者在 3920/3920 行逐字节相同；reward_model.style 在 3920/3920 行都是 rule。因此后续 SFT 应选一个 canonical source，建议使用 reward_model.ground_truth，因为 reward manager 也以它作为 reference。

extra_info.input 是带有历史对话、tool observation 的原始记录，不应直接替代 prompt；extra_info.instruction 是 system instruction 的重复保存，也不是完整的训练 input。

## 2. prompt 与 target 结构统计

### prompt

- 3920/3920 行都是 [system, user] 两条消息。
- system message 含有工具列表、步骤说明和 Output Format。
- 3920/3920 行的 system message 都包含 <think>、<tool_call>、<response> 格式说明。
- SFT input 应保持 prompt 字段原样，再由 Qwen chat template 加上 assistant generation boundary。

### ground_truth

| target 类型 | 数量 | 结构 |
|---|---:|---|
| tool-only | 3447 | 一个 <think> 对，一个 <tool_call> 对 |
| response-only | 473 | 一个 <think> 对，一个 <response> 对 |
| tool + response | 0 | 当前训练集没有同时出现两者的 target |

进一步检查：

- 3920/3920 条都有恰好一个 <think> 和一个 </think>。
- 3447/3447 条 tool target 都有恰好一个 <tool_call> 和一个 </tool_call>。
- 473/473 条 response target 都有恰好一个 <response> 和一个 </response>。
- 当前 strict parser 的正则检查通过 3920/3920。
- 3447 条 tool target 的 tool body 均可按“每行一个 JSON object”解析。
- tool target 的 tool-call 数量分布：1 个 1787 条、2 个 1096 条、3 个 336 条、4 个 173 条、5 个 37 条、6 个 12 条、7 个 4 条、8 个 1 条、9 个 1 条。

这说明原始 dataset 已经含有完整的 format-supervision；当前缺少的不是 label，而是模型从 prompt 生成该协议的能力。

## 3. 完整样本示例：source row 3037

该样本的 data_source=rlla，ability=math。下面保留 parquet 中的 system、user 和 chosen response 内容。

### prompt[0], role=system

    You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

    **Available Tools**
    In your response, you can use the following tools:
    1. Name: recitations_by_juz_number
    Description: Fetches a list of ayah recitations for a specified juz number.
    Parameters: {"recitation_id": {"description": "The ID of the recitation.", "type": "int", "default": ""}, "juz_number": {"description": "The number of the juz for which to fetch ayah recitations.", "type": "int", "default": ""}}
    2. Name: recitations_by_chapter_number
    Description: Fetches a list of ayah recitations for a specific Surah (chapter) based on the given chapter number and recitation ID.
    Parameters: {"chapter_number": {"description": "The chapter (Surah) number for which to fetch the ayah recitations.", "type": "int", "default": ""}, "recitation_id": {"description": "The recitation ID to specify the reciter.", "type": "int", "default": ""}}
    3. Name: getuserbyname
    Description: Retrieves user information from the RapidAPI service based on the provided username.
    Parameters: {"username": {"description": "The name of the user to fetch. Use 'user1' for testing.", "type": "str", "default": "string"}}
    4. Name: numerology
    Description: Calculates the numerology information of a given name by making a request to the horoscope-astrology API using RapidAPI.
    Parameters: {"n": {"description": "The name for which numerology information is to be calculated.", "type": "str", "default": "7"}}

    **Steps for Each Turn**
    1. **Think:** Recall relevant context and analyze the current user goal.
    2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
    3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

    **Output Format**
    [plaintext]
    <think> Your thoughts and reasoning </think>
    <tool_call>
    {"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "... ...": "... ..."}}
    {"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}
    ...
    </tool_call>
    <response> AI's final response </response>

    **Important Notes**
    1. You must always include the <think> field to outline your reasoning. Provide at least one of <tool_call> or <response>. Decide whether to use <tool_call> (possibly multiple times), <response>, or both.
    2. You can invoke multiple tool calls simultaneously in the <tool_call> fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
    3. Refer to the previous dialogue records in the history, including the user's queries, previous <tool_call>, <response>, and any tool feedback noted as <obs> (if exists).

### prompt[1], role=user

    **Dialogue Records History**
    <user> What are the ayah recitations for Surah number 30 as per reciter id 4? Also, get the user information for user3 and look up the numerology information for the name 'Sophia'. </user>

### chosen response

    <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>
    <tool_call>
    {"name": "recitations_by_chapter_number", "parameters": {"chapter_number": 30, "recitation_id": 4}}
    {"name": "getuserbyname", "parameters": {"username": "user3"}}
    {"name": "numerology", "parameters": {"n": "Sophia"}}
    </tool_call>

extra_info.output 与这个 chosen response 完全相同。

## 4. 当前 format contract

对 tool-only target，当前 reward parser 要求严格的结构：

    <think>...</think>
    <tool_call>
    one JSON object per line
    </tool_call>

对 response-only target，要求：

    <think>...</think>
    <response>...</response>

其中换行位置是 contract 的一部分；当前 parser 使用 anchored regular expression，并检查 wrapper 数量。多余的 Markdown fence、额外自然语言、未闭合 tag、重复 <tool_call> 都会导致 format_reward=0。tool correctness parser 还要求 <tool_call> 内每个非空行都是可解析 JSON。

注意：system prompt 展示了同时包含 <tool_call> 和 <response> 的通用格式，但当前 3920 条 chosen target 实际上是二选一，没有同时出现两者的 target。

## 5. 是否需要额外构造 SFT 数据

初始 format warmup 不需要重新发明标签，也不需要手工重写 3920 条 response：

- 原始 reward_model.ground_truth 已经是可用 chosen response。
- 自动 strict-filter 可以验证标签、换行和 JSON，而当前全量 target 已通过。
- 只需从 train 中抽取 500--1000 条，并保留 tool-only、response-only、单 tool、多 tool 的覆盖。
- 当前固定的 16 条诊断样本应从未来 SFT 子集中排除，避免 Before/After validation 泄漏。
- test.parquet 的 80 条也应保持为最终 holdout，不用于 warmup。

人工/规则构造只在需要补充稀有结构时使用，不能作为第一选择：手工 JSON 或合成 tool schema 容易引入与真实 reward parser 不一致的 target。推荐的第一版是“原始 chosen response + 自动过滤 + 分层抽样”，而不是新增 synthetic response。

## 6. 审计结论

可以直接构造 instruction/chat SFT：

    messages = row["prompt"]
    assistant_target = row["reward_model"]["ground_truth"]

用 Qwen 的 chat template 序列化 messages，并把 assistant_target 作为 assistant content。不要把 extra_info.input 当成新的 prompt，不要修改 reward manager，也不要把 SFT 目标改成 Markdown code block 或另一套 tool-call schema。
