#!/usr/bin/env python3
"""
创建1024维度向量数据库的简化脚本
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def create_vector_db():
    """创建1024维度向量数据库"""
    try:
        import chromadb
        from langchain_embedding_utils import langchain_embedding_utils
        
        # 获取配置
        persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'knowledge_base')
        embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', '1024'))
        
        print(f"正在创建 {embedding_dimension} 维度向量数据库...")
        print(f"持久化目录: {persist_directory}")
        print(f"集合名称: {collection_name}")
        
        # 创建持久化客户端
        client = chromadb.PersistentClient(path=persist_directory)
        
        # 删除现有集合（如果存在）
        try:
            client.delete_collection(name=collection_name)
            print(f"✓ 删除现有集合 '{collection_name}'")
        except:
            pass
        
        # 创建新集合，让 ChromaDB 自动检测维度
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ 创建新集合 '{collection_name}'")
        
        # 测试嵌入模型
        test_text = "这是一个测试文本，用于验证嵌入模型是否正常工作。"
        print("正在测试嵌入模型...")
        
        embedding = langchain_embedding_utils.get_embedding(test_text)
        if embedding is None:
            print("✗ 嵌入模型测试失败")
            return False
        
        actual_dimension = len(embedding)
        print(f"✓ 嵌入模型测试成功，实际维度: {actual_dimension}")
        
        if actual_dimension != embedding_dimension:
            print(f"⚠️  维度不匹配: 期望 {embedding_dimension}, 实际 {actual_dimension}")
        
        # 添加测试数据
        sample_texts = [
            "向量数据库是AI应用的重要组成部分。",
            "1024维度的向量可以提供更丰富的语义表示。",
            "RAG技术结合了检索和生成的优势。",
            "ChromaDB是一个轻量级的向量数据库。"
        ]
        
        sample_metadatas = [
            {"source": "test_doc_1", "type": "text", "category": "技术介绍"},
            {"source": "test_doc_2", "type": "text", "category": "技术介绍"},
            {"source": "test_doc_3", "type": "text", "category": "技术介绍"},
            {"source": "test_doc_4", "type": "text", "category": "工具介绍"}
        ]
        
        sample_ids = [f"doc_{i}" for i in range(len(sample_texts))]
        
        # 生成嵌入向量
        print("正在生成测试数据的嵌入向量...")
        embeddings = langchain_embedding_utils.get_embeddings_batch(sample_texts)
        
        if not embeddings:
            print("✗ 嵌入向量生成失败")
            return False
        
        # 添加到集合
        collection.add(
            documents=sample_texts,
            metadatas=sample_metadatas,
            embeddings=embeddings,
            ids=sample_ids
        )
        
        print(f"✓ 成功添加 {len(sample_texts)} 个测试文档")
        print(f"  总文档数量: {collection.count()}")
        
        # 测试查询 - 使用嵌入向量而不是query_texts
        query = "什么是向量数据库？"
        query_embedding = langchain_embedding_utils.get_embedding(query)
        
        if query_embedding:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
            
            print(f"\n✓ 测试查询: '{query}'")
            print("搜索结果:")
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                print(f"  {i+1}. {doc}")
                print(f"     来源: {metadata['source']}")
                print(f"     距离: {distance:.4f}")
        else:
            print("✗ 查询嵌入向量生成失败")
        
        return True
        
    except Exception as e:
        print(f"✗ 创建向量数据库失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== 创建1024维度向量数据库 ===")
    
    if create_vector_db():
        print("\n✓ 1024维度向量数据库创建成功！")
        print("数据库位置: ./chroma_db")
        print("集合名称: knowledge_base")
        print("向量维度: 1024")
        print("嵌入模型: ritrieve-zh")
    else:
        print("\n✗ 向量数据库创建失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()