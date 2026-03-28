graph TD
    A[开始: get_score] --> B{是否有原子事实?}
    
    %% 原子事实生成阶段
    B -- 否 --> C[AtomicFactGenerator.run]
    C --> C1[NLTK 分句 & 修复]
    C1 --> C2[BM25 检索 Few-shot 示例]
    C2 --> C3[调用 OpenAI InstructGPT]
    C3 --> C4[后处理: 实体链接 & 修正]
    C4 --> D[得到原子事实列表]
    
    B -- 是 --> D
    
    %% 评分阶段
    D --> E[遍历原子事实: _get_score]
    E --> F[Retrieval.get_passages]
    F --> F1[DocDB.get_text_from_title]
    F1 --> F2[BM25 / GTR 检索相关段落]
    F2 --> G[构建验证 Prompt]
    
    G --> H{选择验证模型}
    H -- ChatGPT --> I1[调用 OpenAI API]
    H -- Llama --> I2[本地运行 Llama 获取 Logits]
    
    I1 --> J[解析 True/False 结果]
    I2 --> J
    
    J --> K{开启 NPM?}
    K -- 是 --> L[NPM.get_probability 进一步验证]
    L --> M[最终判定: is_supported]
    K -- 否 --> M
    
    %% 结果汇总
    M --> N[汇总所有事实得分]
    N --> O[计算长度惩罚 Gamma Penalty]
    O --> P[输出 FActScore 结果]
    P --> Q[结束]


### 核心逻辑说明
1. 原子事实拆解 : 并不是简单的分句，而是利用 LLM 将复杂句子拆解为多个独立的、可验证的陈述（Atomic Facts）。
2. 两级检索 : 首先根据 Topic （如人名）从数据库中锁定对应的完整文档，然后在文档内部利用 BM25 或 GTR 检索与特定原子事实最相关的 5 个段落。
3. 多重验证 :
   - LLM 验证 : 通过 Context + Fact 构建 Prompt，让模型判断真伪。
   - NPM 辅助 : 如果模型判断为真，还可以通过非参数化概率模型（NPM）进行二次确认，增强结果的可靠性。
4. 长度惩罚 : FActScore 引入了 Gamma 参数，如果生成的事实数量太少，会降低最终得分，以防止模型通过生成简短、保守的内容来刷分。