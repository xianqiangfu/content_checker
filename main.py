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

    def generate_atomic_facts(self, sentence: str) -> List[str]:
        """
        2. 对每一句话，生成一个或几个原子 fact (优化后的 Few-shot 逻辑)
        参考 FActScore 的实现，使用 Few-shot 提示词来提高拆解的准确性。
        """
        # 定义 Few-shot 示例
        demos = """请将以下句子拆解为若干个独立的原子事实 (Atomic Facts)。
每个事实应该是一个简单的陈述句，包含一个独立的信息点。

示例 1:
句子: "张三是一名出生于 1990 年的软件工程师，目前在阿里巴巴工作。"
原子事实:
- 张三是一名软件工程师。
- 张三出生于 1990 年。
- 张三目前在阿里巴巴工作。

示例 2:
句子: "他不仅擅长 Python 开发，还精通 Java，并且是项目的负责人。"
原子事实:
- 他擅长 Python 开发。
- 他精通 Java。
- 他是项目的负责人。

待处理句子: "{sentence}"
原子事实:"""
        
        prompt = demos.format(sentence=sentence)
        response = self.llm.invoke(prompt)
        
        # 解析输出
        facts = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line.startswith("-"):
                fact = line.strip("- ").strip()
                if fact:
                    facts.append(fact)
        
        # 兜底逻辑：如果模型没有按格式输出，尝试进行简单的清洗或直接返回原句
        if not facts:
            # 尝试按行分割
            lines = [l.strip() for l in response.content.split('\n') if len(l.strip()) > 5 and ":" not in l]
            facts = lines if lines else [sentence]
            
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
