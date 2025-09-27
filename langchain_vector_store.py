#!/usr/bin/env python3
"""
基于 LangChain 的向量存储管理器
支持多种文档格式，包括 Excel
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader
)
from langchain.schema import Document
from langchain_embedding_utils import langchain_embedding_utils

load_dotenv()

class LangChainVectorStore:
    def __init__(self):
        self.persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        self.collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'knowledge_base')
        
        self.vectorstore = None
        self.text_splitter = None
        self._initialize_vectorstore()
        self._initialize_text_splitter()
    
    def _initialize_vectorstore(self):
        """初始化向量存储"""
        try:
            if langchain_embedding_utils.embeddings is None:
                print("✗ 嵌入模型未初始化，无法初始化向量存储")
                return
            
            # 尝试获取现有集合，如果不存在则创建新集合
            try:
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=langchain_embedding_utils.embeddings,
                    persist_directory=self.persist_directory
                )
                print(f"✓ LangChain 向量存储初始化成功")
                print(f"  持久化目录: {self.persist_directory}")
                print(f"  集合名称: {self.collection_name}")
            except Exception as inner_e:
                # 如果集合已存在，直接连接到现有集合
                if "already exists" in str(inner_e):
                    print(f"✓ 集合 '{self.collection_name}' 已存在，连接到现有集合")
                    self.vectorstore = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=langchain_embedding_utils.embeddings,
                        persist_directory=self.persist_directory
                    )
                else:
                    raise inner_e
        except Exception as e:
            print(f"✗ 向量存储初始化失败: {e}")
    
    def _initialize_text_splitter(self):
        """初始化文本分割器"""
        chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _get_loader(self, file_path: str, file_type: str):
        """根据文件类型获取对应的加载器"""
        try:
            if file_type == 'txt':
                return TextLoader(file_path, encoding='utf-8')
            elif file_type == 'pdf':
                return PyPDFLoader(file_path)
            elif file_type == 'docx':
                return Docx2txtLoader(file_path)
            elif file_type == 'pptx':
                return UnstructuredPowerPointLoader(file_path)
            elif file_type in ['xlsx', 'xls']:
                return UnstructuredExcelLoader(file_path)
            elif file_type == 'csv':
                return CSVLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
        except ImportError as e:
            print(f"✗ 加载器导入失败: {e}")
            return None
    
    def load_document(self, file_path: str, metadata: Optional[Dict] = None) -> List[Document]:
        """加载文档并分割成块"""
        try:
            file_type = file_path.lower().split('.')[-1]
            loader = self._get_loader(file_path, file_type)
            
            if loader is None:
                # 如果没有对应的加载器，尝试读取为文本
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents = [Document(page_content=content, metadata=metadata or {})]
                except Exception as e:
                    print(f"✗ 文件读取失败: {e}")
                    return []
            else:
                # 使用对应的加载器
                documents = loader.load()
                
                # 添加元数据
                if metadata:
                    for doc in documents:
                        doc.metadata.update(metadata)
                
                # 为每个文档添加文件信息
                for doc in documents:
                    doc.metadata.update({
                        'source': os.path.basename(file_path),
                        'file_type': file_type,
                        'file_path': file_path
                    })
            
            # 文本分割
            if self.text_splitter and len(documents) > 0:
                split_docs = self.text_splitter.split_documents(documents)
                print(f"✓ 文档分割完成: {len(documents)} -> {len(split_docs)} 个块")
                return split_docs
            else:
                return documents
                
        except Exception as e:
            print(f"✗ 文档加载失败: {e}")
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """添加文档到向量存储"""
        if self.vectorstore is None:
            print("✗ 向量存储未初始化")
            return False
        
        if not documents:
            print("✗ 没有文档可添加")
            return False
        
        try:
            # 添加文档
            self.vectorstore.add_documents(documents)
            print(f"✓ 成功添加 {len(documents)} 个文档到向量存储")
            
            # 注意：在新版本的 LangChain Chroma 中，persist() 方法已被移除
            # 数据会自动持久化到磁盘
            print(f"✓ 向量存储已自动持久化到 {self.persist_directory}")
            
            return True
        except Exception as e:
            print(f"✗ 添加文档失败: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """相似度搜索"""
        if self.vectorstore is None:
            print("✗ 向量存储未初始化")
            return []
        
        try:
            # 执行相似度搜索
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # 转换结果格式
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return results
        except Exception as e:
            print(f"✗ 相似度搜索失败: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """带分数的相似度搜索"""
        if self.vectorstore is None:
            print("✗ 向量存储未初始化")
            return []
        
        try:
            # 执行带分数的相似度搜索
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # 转换结果格式
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return results
        except Exception as e:
            print(f"✗ 相似度搜索失败: {e}")
            return []
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        if self.vectorstore is None:
            return 0
        
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
    
    def clear_collection(self) -> bool:
        """清空集合"""
        if self.vectorstore is None:
            return False
        
        try:
            # 删除并重新创建集合
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=langchain_embedding_utils.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"✓ 集合已清空: {self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ 清空集合失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取向量存储状态"""
        return {
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "document_count": self.get_document_count(),
            "embedding_status": langchain_embedding_utils.get_model_info(),
            "vectorstore_status": "initialized" if self.vectorstore else "failed"
        }

# 全局向量存储实例
langchain_vector_store = LangChainVectorStore()
