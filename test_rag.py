import unittest
from unittest.mock import MagicMock, patch
from rag_system import RAGSystem

class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        # 准备测试数据
        self.test_data = [
            "LangChain 是一个 LLM 框架。",
            "Ollama 可以在本地运行大模型。",
            "FAISS 是向量库。",
            "RAG 代表检索增强生成。"
        ]
        
    @patch('rag_system.OllamaEmbeddings')
    @patch('rag_system.ChatOllama')
    @patch('rag_system.FAISS')
    def test_build_vector_store(self, mock_faiss, mock_chat_ollama, mock_ollama_embeddings):
        # 模拟 Embedding 维度
        mock_ollama_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 4
        
        # 模拟 FAISS
        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store
        
        # 初始化 RAG 系统
        rag = RAGSystem()
        rag.build_vector_store(self.test_data)
        
        # 验证 FAISS.from_documents 被调用
        self.assertTrue(mock_faiss.from_documents.called)
        self.assertIsNotNone(rag.vector_store)
        self.assertIsNotNone(rag.retriever)

    @patch('rag_system.OllamaEmbeddings')
    @patch('rag_system.ChatOllama')
    @patch('rag_system.FAISS')
    def test_query(self, mock_faiss, mock_chat_ollama, mock_ollama_embeddings):
        # 模拟返回值
        mock_response = MagicMock()
        mock_response.content = "RAG 是检索增强生成。"
        mock_chat_ollama.return_value.invoke.return_value = mock_response
        
        # 模拟检索器
        mock_doc = MagicMock()
        mock_doc.page_content = "RAG 代表检索增强生成。"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vector_store
        
        # 运行查询
        rag = RAGSystem()
        rag.build_vector_store(self.test_data)
        result = rag.query("什么是 RAG？")
        
        # 验证结果
        self.assertEqual(result, "RAG 是检索增强生成。")

if __name__ == '__main__':
    unittest.main()
