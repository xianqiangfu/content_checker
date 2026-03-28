import os
import nltk
from typing import List
from langchain_ollama import ChatOllama
from rag_system import RAGSystem

# 确保 NLTK 的 punkt 资源已下载
nltk.download('punkt', quiet=True)

class HallucinationChecker:
    def __init__(self, 
                 model_name: str = "llama3", 
                 base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model_name = model_name
        
        # 初始化 LLM (本地 Ollama)
        self.llm = ChatOllama(
            model=model_name,
            base_url=self.base_url
        )
        
        # 初始化 RAG 系统
        self.rag = RAGSystem(model_name=model_name, base_url=base_url)
        
    def build_knowledge_base(self, texts: List[str]):
        """构建 RAG 知识库"""
        self.rag.build_vector_store(texts)

    def split_into_sentences(self, text: str) -> List[str]:
        """1. 将大模型生成的答案切分为一句一句话"""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def _clean_atomic_fact(self, fact: str) -> str:
        """清洗提取出的原子事实，确保其是一个简洁、完整的陈述句"""
        fact = fact.strip("- ").strip()
        if not fact:
            return ""
        # 确保以句号结尾
        if not fact.endswith(("。", ".", "！", "!", "？", "?")):
            fact += "。"
        return fact

    def generate_atomic_facts(self, sentence: str) -> List[str]:
        """
        2. 对每一句话，生成一个或几个原子 fact (优化后的 Few-shot 逻辑)
        参考 FActScore (atomic_facts.py) 的核心逻辑：
        - 使用更广泛的 Few-shot 示例来引导模型进行信息拆解
        - 强调原子事实的独立性、简洁性和完整性
        - 添加后处理清洗逻辑
        """
        system_prompt = "你是一个专业的文本分析助手，擅长将复杂的句子拆解为最基础的、独立的原子事实 (Atomic Facts)。"
        
        # 定义 Few-shot 示例 (参考 FActScore 的 demons 逻辑)
        demos = [
            {
                "sentence": "张三是一名出生于 1990 年的软件工程师，目前在阿里巴巴工作。",
                "facts": [
                    "张三是一名软件工程师。",
                    "张三出生于 1990 年。",
                    "张三目前在阿里巴巴工作。"
                ]
            },
            {
                "sentence": "李四不仅精通 Python，还获得过 2023 年的最佳开发者奖项。",
                "facts": [
                    "李四精通 Python。",
                    "李四获得过最佳开发者奖项。",
                    "该奖项是在 2023 年获得的。"
                ]
            }
        ]

        # 构建 Prompt
        prompt_content = "请将以下待处理句子拆解为若干个独立的原子事实。每个事实必须是简单的陈述句，包含且仅包含一个独立的信息点。请直接以列表形式输出，每行一个，以 '-' 开头。\n\n"
        for i, demo in enumerate(demos):
            prompt_content += f"示例 {i+1}:\n句子: \"{demo['sentence']}\"\n原子事实:\n"
            for f in demo['facts']:
                prompt_content += f"- {f}\n"
            prompt_content += "\n"

        prompt_content += f"待处理句子: \"{sentence}\"\n原子事实:"
        
        # 调用大模型 (使用 ChatOllama 的 invoke)
        response = self.llm.invoke(f"{system_prompt}\n\n{prompt_content}")
        
        # 优化解析逻辑：使用正则提取以 - 或 • 或 1. 开头的行
        import re
        raw_content = response.content
        
        facts = []
        for line in raw_content.split('\n'):
            line = line.strip()
            # 匹配常见的列表符号
            match = re.match(r'^[-*•\d+\.]\s*(.*)', line)
            if match:
                fact_text = match.group(1).strip()
                cleaned = self._clean_atomic_fact(fact_text)
                if cleaned:
                    facts.append(cleaned)
            elif line and not any(line.startswith(s) for s in ["示例", "句子", "原子事实", "待处理"]):
                # 兜底：如果是普通非空行且不是 Prompt 中的关键词，也尝试清洗
                cleaned = self._clean_atomic_fact(line)
                if cleaned and len(cleaned) > 5:
                    facts.append(cleaned)
        
        # 最终兜底：如果还是没提取出来，返回原句
        if not facts:
            facts = [self._clean_atomic_fact(sentence)]
            
        return facts


    def retrieve_context(self, sentence: str) -> str:
        """3. 将这句话在 rag 中查找相关片段"""
        # 尝试使用带重排序的检索器获取相关文档
        try:
            # 默认尝试使用 rerank (LLMChainExtractor)
            return self.rag.get_context(sentence, use_rerank=True)
        except Exception:
            # 降级到普通检索
            if self.rag.retriever:
                docs = self.rag.retriever.invoke(sentence)
                return "\n\n".join(doc.page_content for doc in docs)
        return ""

    def judge_hallucination(self, sentence: str, facts: List[str], context: str) -> str:
        """4. 将第二步、第三步的结果送入大模型进行判断"""
        facts_str = "\n".join([f"{i+1}. {f}" for i, f in enumerate(facts)])
        
        prompt = f"""请扮演一个严格的事实核查员。根据提供的上下文 (Context)，对给出的原子事实 (Atomic Facts) 进行逐条核实。

[上下文]
{context}

[待核实的原子事实]
{facts_str}

[核实要求]
1. 请对每个事实给出判断：【正确】、【错误】或【不确定】。
2. 简要说明理由，特别是对于【错误】或【不确定】的事实，请指出上下文中的矛盾点或缺失点。
3. 如果上下文完全没有提及相关信息，请标记为【不确定】。

请以 Markdown 表格形式输出核实结果，表格列为：序号 | 原子事实 | 判定结果 | 理由说明

核实结果:"""
        response = self.llm.invoke(prompt)
        return response.content

    def check_answer(self, answer: str):
        """运行完整流程"""
        print(f"\n{'='*20} 开始幻觉检测 {'='*20}")
        print(f"原始回答: {answer}\n")
        
        # 1. 分句
        sentences = self.split_into_sentences(answer)
        print(f"共识别到 {len(sentences)} 个句子。\n")
        
        for i, sentence in enumerate(sentences):
            print(f"[{i+1}/{len(sentences)}] 处理句子: \"{sentence}\"")
            
            # 2. 生成原子事实
            facts = self.generate_atomic_facts(sentence)
            print(f"   > 提取原子事实 ({len(facts)}个):")
            for f in facts:
                print(f"     - {f}")
            
            # 3. 检索 RAG 片段
            context = self.retrieve_context(sentence)
            # print(f"   > 检索到的上下文 (前 100 字): {context[:100].replace('\\n', ' ')}...")
            
            # 4. 幻觉判断
            judgment = self.judge_hallucination(sentence, facts, context)
            print(f"\n   > 判定结果:\n{judgment}\n")
            print("-" * 60)
        
        print(f"{'='*20} 检测结束 {'='*20}\n")

if __name__ == "__main__":
    # 示例运行
    checker = HallucinationChecker()
    
    # 假设 RAG 系统中有一些知识
    knowledge = [
        "张三是一名出生于 1990 年的软件工程师，目前在阿里巴巴工作。",
        "李四是张三的同事，擅长 Python 和 Java 开发。",
        "阿里巴巴的总部位于中国杭州。"
    ]
    checker.build_knowledge_base(knowledge)
    
    # 待检测的大模型答案
    test_answer = "张三是阿里巴巴的工程师。他出生于 1995 年，擅长写 C++ 代码。"
    
    checker.check_answer(test_answer)
