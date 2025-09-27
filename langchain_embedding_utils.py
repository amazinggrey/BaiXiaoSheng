#!/usr/bin/env python3
"""
基于 LangChain 的向量嵌入工具模块
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class LangChainEmbeddingUtils:
    def __init__(self):
        self.model_type = os.getenv('EMBEDDING_MODEL_TYPE', 'ollama')
        self.model_name = os.getenv('EMBEDDING_MODEL_NAME', 'nomic-embed-text:v1.5')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', '1024'))
        
        self.embeddings = None
        self.text_splitter = None
        self._initialize_embeddings()
        self._initialize_text_splitter()
    
    def _initialize_embeddings(self):
        """初始化 LangChain 嵌入模型"""
        try:
            if self.model_type == 'sentence-transformers':
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'}
                )
                print(f"✓ LangChain HuggingFace 嵌入模型加载成功: {self.model_name}")
            elif self.model_type == 'ollama':
                self.embeddings = OllamaEmbeddings(
                    model=self.model_name,
                    base_url=self.ollama_base_url
                )
                print(f"✓ LangChain Ollama 嵌入模型加载成功: {self.model_name}")
            elif self.model_type == 'openai':
                api_key = os.getenv('OPENAI_API_KEY', '')
                if api_key:
                    self.embeddings = OpenAIEmbeddings(
                        model=self.model_name,
                        openai_api_key=api_key
                    )
                    print(f"✓ LangChain OpenAI 嵌入模型加载成功: {self.model_name}")
                else:
                    print("✗ OpenAI API Key 未配置")
            else:
                print(f"✗ 不支持的嵌入模型类型: {self.model_type}")
        except Exception as e:
            print(f"✗ 嵌入模型初始化失败: {e}")
    
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
        print(f"✓ 文本分割器初始化成功: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量嵌入"""
        if self.embeddings is None:
            print("✗ 嵌入模型未初始化")
            return None
        
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"✗ 嵌入生成失败: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的向量嵌入"""
        if self.embeddings is None:
            print("✗ 嵌入模型未初始化")
            return []
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            print(f"✗ 批量嵌入生成失败: {e}")
            return []
    
    def split_text(self, text: str) -> List[str]:
        """文本分块"""
        if self.text_splitter is None:
            return [text]
        
        try:
            chunks = self.text_splitter.split_text(text)
            return chunks
        except Exception as e:
            print(f"✗ 文本分割失败: {e}")
            return [text]
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "status": "initialized" if self.embeddings else "failed"
        }

# 全局嵌入工具实例
langchain_embedding_utils = LangChainEmbeddingUtils()
