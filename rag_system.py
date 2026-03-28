import os
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA

class RAGSystem:
    def __init__(self, 
                 model_name: str = "llama3", 
                 embedding_model: str = "llama3", 
                 base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model_name = model_name
        
        # 1. 初始化 Embedding (本地 Ollama)
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=self.base_url
        )
        
        # 2. 初始化 LLM (本地 Ollama)
        self.llm = ChatOllama(
            model=model_name,
            base_url=self.base_url
        )
        
        self.vector_store = None
        self.retriever = None

    def build_vector_store(self, texts: List[str]):
        """建库：将文本分块并存储到 FAISS 向量数据库中"""
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        documents = [Document(page_content=t) for t in texts]
        splits = text_splitter.split_documents(documents)
        
        # 创建向量库
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("向量库构建完成。")

    def setup_rerank_retriever(self):
        """配置重排序 (使用 LLM 作为 Reranker)"""
        # 这里使用 LLMChainExtractor 作为简单的重排序/压缩器
        # 如果需要更专业的重排序，通常使用 CrossEncoder (如 BGE-Reranker)
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
        print("重排序检索器配置完成。")

    def query(self, question: str, use_rerank: bool = False):
        """查询：检索并生成回答"""
        if not self.retriever:
            raise ValueError("请先构建向量库。")

        retriever = self.compression_retriever if use_rerank and hasattr(self, 'compression_retriever') else self.retriever
        
        # 定义提示词模板
        template = """你是一个专业的助手。请根据以下提供的上下文回答问题。
如果你不知道答案，就直说不知道，不要编造。

上下文:
{context}

问题: {question}

回答:"""
        prompt = ChatPromptTemplate.from_template(template)

        # 构建 RAG 链
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )

        response = rag_chain.invoke(question)
        return response.content

if __name__ == "__main__":
    # 示例运行
    rag = RAGSystem()
    
    # 准备一些测试数据
    test_data = [
        "LangChain 是一个用于构建 LLM 应用的框架。",
        "Ollama 是一个在本地运行大型语言模型的工具。",
        "RAG (Retrieval-Augmented Generation) 通过检索外部知识来增强生成模型的能力。",
        "FAISS 是 Meta 开发的一个用于高效相似性搜索的库。"
    ]
    
    rag.build_vector_store(test_data)
    rag.setup_rerank_retriever()
    
    q = "什么是 RAG？"
    print(f"问题: {q}")
    print(f"回答: {rag.query(q, use_rerank=True)}")
