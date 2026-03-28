import json
import time
import os

import sqlite3
import numpy as np
import pickle as pkl

from rank_bm25 import BM25Okapi

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256

class DocDB(object):
    """
    基于 SQLite 的文档存储，用于快速获取文档文本。
    """

    def __init__(self, db_path=None, data_path=None):
        """
        初始化数据库。如果数据库不存在，则从 data_path 构建。
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        
        if len(cursor.fetchall())==0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print (f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def build_db(self, db_path, data_path):
        """从 JSONL 文件构建 SQLite 数据库。"""
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        
        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text)==str:
                    text = [text]
                
                # 将文本切分为固定长度的 Passage (使用 Roberta Tokenizer)
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip())>0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset+MAX_LENGTH])
                            offset += MAX_LENGTH
                
                psgs = [tokenizer.decode(tokens) for tokens in passages if np.sum([t not in [0, 2] for t in tokens])>0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)

        self.connection.commit()

    def get_text_from_title(self, title):
        """根据标题（Topic）从数据库中获取所有相关的段落。"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert results is not None and len(results)==1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        return results

class Retrieval(object):
    """
    检索类，支持 BM25 和基于嵌入 (GTR) 的检索。
    """

    def __init__(self, db, cache_path, embed_cache_path,
                 retrieval_type="gtr-t5-large", batch_size=None):
        """
        初始化检索器。
        :param db: DocDB 实例。
        :param cache_path: 检索结果的缓存路径。
        :param embed_cache_path: 嵌入向量的缓存路径。
        :param retrieval_type: 检索类型 ("bm25" 或 "gtr-...")。
        """
        self.db = db
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type=="bm25" or retrieval_type.startswith("gtr-")
        
        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0

    def load_encoder(self):
        """加载 SentenceTransformer 模型。"""
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None
    
    def get_passages(self, topic, question, k):
        """
        根据主题和问题，检索最相关的 k 个段落。
        """
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query
        
        if cache_key not in self.cache:
            # 首先从 DB 获取该 Topic 下的所有 Passage
            passages = self.db.get_text_from_title(topic)
            # 然后在该 Topic 内部进行检索
            if self.retrieval_type=="bm25":
                self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
            else:
                self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
            self.add_n += 1
        
        return self.cache[cache_key]

    def load_cache(self):
        """
        从磁盘加载检索和嵌入的缓存。
        """
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}
    
    def save_cache(self):
        """
        保存缓存到磁盘，采用合并模式防止多进程冲突。
        """
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)
            
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        
        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)
            
            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_bm25_passages(self, topic, query, passages, k):
        """
        在给定的段落集合中执行 BM25 检索。
        """
        if topic in self.embed_cache:
            bm25 = self.embed_cache[topic]
        else:
            # 如果缓存中没有该主题的 BM25 索引，则即时创建
            bm25 = BM25Okapi([psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages])
            self.embed_cache[topic] = bm25
            self.add_n_embed += 1
        
        # 计算 BM25 得分并返回前 k 个段落
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_gtr_passages(self, topic, retrieval_query, passages, k):
        """
        使用 GTR 嵌入模型执行向量相似度检索。
        """
        if self.encoder is None:
            self.load_encoder()
        
        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            # 对所有段落进行编码并存入缓存
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
            passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device)
            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1
            
        # 编码查询语句并计算与段落向量的内积
        query_vectors = self.encoder.encode([retrieval_query], 
                                            batch_size=self.batch_size,
                                            device=self.encoder.device)[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]



        
        


