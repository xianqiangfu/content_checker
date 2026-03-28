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
            },
            {
                "sentence": "该公司的总部位于北京，由王五在 2005 年创立。",
                "facts": [
                    "该公司的总部位于北京。",
                    "该公司是由王五创立的。",
                    "该公司创立于 2005 年。"
                ]
            }
        ]

        # 构建 Prompt
        prompt_content = "请将以下待处理句子拆解为若干个独立的原子事实。每个事实必须是简单的陈述句，包含且仅包含一个独立的信息点。\n\n"
        for i, demo in enumerate(demos):
            prompt_content += f"示例 {i+1}:\n句子: \"{demo['sentence']}\"\n原子事实:\n"
            for f in demo['facts']:
                prompt_content += f"- {f}\n"
            prompt_content += "\n"

        prompt_content += f"待处理句子: \"{sentence}\"\n原子事实:"
        
        # 调用大模型 (使用 ChatOllama 的 invoke)
        response = self.llm.invoke(f"{system_prompt}\n\n{prompt_content}")
        
        # 优化解析逻辑 (参考 atomic_facts.py 的 text_to_sentences 思想)
        raw_lines = response.content.split('\n')
        facts = []
        for line in raw_lines:
            cleaned = self._clean_atomic_fact(line)
            if cleaned and (line.strip().startswith("-") or line.strip().startswith("•")):
                facts.append(cleaned)
        
        # 兜底逻辑：如果模型没有按 "-" 格式输出，尝试提取所有非空行
        if not facts:
            facts = [self._clean_atomic_fact(l) for l in raw_lines if len(l.strip()) > 5 and ":" not in l]
            facts = [f for f in facts if f]

        # 最终兜底：如果还是没提取出来，返回原句
        if not facts:
            facts = [self._clean_atomic_fact(sentence)]
            
        return facts


    def retrieve_context(self, sentence: str) -> str:
        """3. 将这句话在 rag 中查找相关片段"""
        # 使用 RAG 系统的检索器获取相关文档
        if self.rag.retriever:
            docs = self.rag.retriever.invoke(sentence)
            return "\n\n".join(doc.page_content for doc in docs)
        return ""

    def judge_hallucination(self, sentence: str, facts: List[str], context: str) -> str:
        """4. 将第二步、第三步的结果送入大模型进行判断"""
        facts_str = "\n".join([f"- {f}" for f in facts])
        
        prompt = f"""请根据提供的上下文 (Context)，判断以下原子事实 (Atomic Facts) 是否包含幻觉。

上下文:
{context}

原子事实:
{facts_str}

对于每一个原子事实，请按照以下标准进行判断：
1. 有幻觉，生成内容不正确；
2. 不确定；
3. 无幻觉，生成内容正确。

最后给出该句子的综合评估。

判断结果:"""
        response = self.llm.invoke(prompt)
        return response.content

    def check_answer(self, answer: str):
        """运行完整流程"""
        print(f"--- 开始幻觉检测 ---")
        print(f"原始回答: {answer}\n")
        
        # 1. 分句
        sentences = self.split_into_sentences(answer)
        
        for i, sentence in enumerate(sentences):
            print(f"[{i+1}] 处理句子: {sentence}")
            
            # 2. 生成原子事实
            facts = self.generate_atomic_facts(sentence)
            print(f"   - 提取的原子事实: {facts}")
            
            # 3. 检索 RAG 片段
            context = self.retrieve_context(sentence)
            print(f"   - 检索到的上下文 (前 100 字): {context[:100]}...")
            
            # 4. 幻觉判断
            judgment = self.judge_hallucination(sentence, facts, context)
            print(f"   - 判定结果:\n{judgment}\n")
            print("-" * 50)

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
