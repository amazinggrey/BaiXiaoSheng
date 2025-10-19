import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
# 使用 LangGraph 的持久化功能替代旧的 memory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid
from transformers import pipeline

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """基于 LangChain 的 RAG 系统实现"""
    
    def __init__(self, 
                 collection_name: str = "knowledge_base",
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-3.5-turbo",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 memory_type: str = "buffer",
                 memory_k: int = 5):
        """
        初始化 RAG 系统
        
        Args:
            collection_name: ChromaDB 集合名称
            embedding_model: OpenAI 嵌入模型
            llm_model: OpenAI LLM 模型
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
            memory_type: 记忆类型 ("buffer", "summary", "window")
            memory_k: 记忆窗口大小（仅对window类型有效）
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.memory_type = memory_type
        self.memory_k = memory_k
        
        # 初始化嵌入模型 - 根据配置选择合适的嵌入模型
        embedding_type = os.getenv("EMBEDDING_MODEL_TYPE", "ollama")
        
        if embedding_type == "ollama":
            from langchain_ollama import OllamaEmbeddings
            self.embedding_model = OllamaEmbeddings(
                model=os.getenv("OLLAMA_EMBEDDING_MODEL"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
        elif embedding_type == "openai":
            self.embedding_model = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE")
            )
        elif embedding_type == 'sentence-transformers':
            from  langchain_huggingface import HuggingFaceEmbeddings
            self.embedding_model = HuggingFaceEmbeddings(
                    model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
                    model_kwargs={'device': 'cpu'}
                )
        else:
            # 默认使用 OpenAI 兼容接口
            self.embedding_model = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
                openai_api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL")
            )
        
        # 初始化 LLM - 根据配置选择合适的LLM模型
        llm_type = os.getenv("LLM_TYPE", "openai")
        
        if llm_type == "ollama":
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=os.getenv("OLLAMA_LLM_MODEL", "llama2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0
            )
        elif llm_type == "openai":
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                base_url=os.getenv("LLM_BASE_URL"),
                temperature=0,
                openai_api_key=os.getenv("LLM_API_KEY")
            )
        elif llm_type == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint
            self.llm = HuggingFaceEndpoint(
                repo_id=os.getenv("HUGGINGFACE_MODEL", "microsoft/DialoGPT-medium"),
                task="text-generation",
                temperature=0,
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
            )
        elif llm_type == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                temperature=0,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif llm_type == "azure":
            from langchain_openai import AzureChatOpenAI
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("OPENAI_API_VERSION", "2023-12-01-preview"),
                temperature=0,
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        else:
            # 默认使用 OpenAI 兼容接口
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                base_url=os.getenv("LLM_BASE_URL"),
                temperature=0,
                openai_api_key=os.getenv("LLM_API_KEY")
            )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # 初始化向量存储
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.vector_store = None
        self.retriever = None
        
        # 检查并创建集合
        self._initialize_collection()
        
        # 初始化记忆系统
        self._initialize_memory()
        
        # 创建 RAG 链
        self._create_rag_chain()
    
    def _initialize_collection(self):
        """初始化 ChromaDB 集合"""
        try:
            # 尝试设置向量存储，如果集合已存在会自动连接
            try:
                self.vector_store = Chroma(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                )
                logger.info(f"✓ 成功连接到集合: {self.collection_name}")
            except Exception as inner_e:
                # 如果集合已存在，直接连接到现有集合
                if "already exists" in str(inner_e):
                    logger.info(f"✓ 集合 '{self.collection_name}' 已存在，直接连接")
                    self.vector_store = Chroma(
                        client=self.client,
                        collection_name=self.collection_name,
                        embedding_function=self.embedding_model,
                    )
                else:
                    raise inner_e
            
            # 设置检索器
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            logger.info("向量存储初始化完成")
            
        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            raise
    
    def _create_rag_chain(self):
        """创建 RAG 处理链"""
        # 定义提示模板
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是Gientech公司专业的知识库助手。根据给定的上下文和对话历史，回答用户的问题。\n\n上下文：\n\n{context}\n\n{system_message}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ])
        
        # 创建上下文处理链
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(x["context"]),
                chat_history=lambda x: self._format_chat_history(x["chat_history"]),
            )
            | contextualize_q_prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("RAG 链创建完成")
    
    def _initialize_memory(self):
        """初始化记忆系统 - 使用 LangGraph 方式"""
        try:
            # 使用 InMemoryChatMessageHistory 存储消息历史
            self.message_history = InMemoryChatMessageHistory()
            
            # 为不同的记忆类型创建消息处理器
            if self.memory_type == "buffer":
                # 完整对话记忆 - 直接存储所有消息
                self.message_processor = self._process_buffer_memory
            elif self.memory_type == "window":
                # 滑动窗口记忆 - 只保留最近的 K 条消息
                self.message_processor = self._process_window_memory
            elif self.memory_type == "summary":
                # 摘要记忆 - 需要定期生成摘要
                self.message_processor = self._process_summary_memory
                self.conversation_summary = ""
            else:
                # 默认使用缓冲记忆
                self.message_processor = self._process_buffer_memory
            
            logger.info(f"✓ 记忆系统初始化成功: {self.memory_type}")
            
        except Exception as e:
            logger.error(f"记忆系统初始化失败: {e}")
            # 降级为简单字典记忆
            self.message_history = None
            self.simple_memory = []
    
    def _process_buffer_memory(self, messages):
        """处理缓冲记忆 - 返回所有消息"""
        return messages
    
    def _process_window_memory(self, messages):
        """处理窗口记忆 - 只返回最近的 K 条消息"""
        return messages[-self.memory_k * 2:] if len(messages) > self.memory_k * 2 else messages
    
    def _process_summary_memory(self, messages):
        """处理摘要记忆 - 返回摘要和最近消息"""
        # 如果消息数量超过阈值，生成摘要
        if len(messages) > 8:  # 阈值可以根据需要调整
            # 生成摘要
            recent_messages = messages[-4:]  # 保留最近4条消息
            older_messages = messages[:-4]
            
            if older_messages:
                # 构建对话内容
                conversation_lines = []
                for msg in older_messages:
                    msg_type = "human" if isinstance(msg, HumanMessage) else "ai"
                    conversation_lines.append(f"{msg_type}: {msg.content}")
                
                # 创建摘要提示
                summary_prompt = f"""请将以下对话内容总结为一个简洁的摘要：

对话内容：
{"\\n".join(conversation_lines)}

摘要："""
                
                summary_response = self.llm.invoke(summary_prompt)
                self.conversation_summary = summary_response.content
                
                # 创建包含摘要的消息列表
                summary_message = AIMessage(content=f"对话摘要: {self.conversation_summary}")
                return [summary_message] + recent_messages
            else:
                return recent_messages
        else:
            return messages
    
    def _format_docs(self, docs: List[Document]) -> str:
        """格式化检索到的文档"""
        return "\n\n".join(
            f"文档 {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(docs)
        )
    
    def _format_chat_history(self, chat_history) -> List:
        """格式化聊天历史"""
        if not chat_history:
            return []
        
        # 如果已经是消息对象列表，直接返回
        if all(isinstance(msg, (HumanMessage, AIMessage)) for msg in chat_history):
            return chat_history
        
        # 如果是元组列表，转换为消息对象
        formatted = []
        for item in chat_history:
            if isinstance(item, tuple) and len(item) == 2:
                role, message = item
                if role == "user":
                    formatted.append(HumanMessage(content=message))
                else:
                    formatted.append(AIMessage(content=message))
            else:
                # 如果已经是消息对象，直接添加
                if isinstance(item, (HumanMessage, AIMessage)):
                    formatted.append(item)
        
        return formatted
    
    def add_documents(self, 
                     texts: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        添加文档到向量存储
        
        Args:
            texts: 文本内容列表
            metadatas: 元数据列表（可选）
            
        Returns:
            bool: 添加是否成功
        """
        try:
            # 创建文档对象
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                documents.append(Document(page_content=text, metadata=metadata))
            
            # 分割文档
            split_docs = self.text_splitter.split_documents(documents)
            
            # 添加到向量存储
            self.vector_store.add_documents(split_docs)
            logger.info(f"成功添加 {len(texts)} 个文档，分割为 {len(split_docs)} 个块")
            
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def add_document_from_file(self, 
                              file_path: str, 
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        从文件添加文档
        
        Args:
            file_path: 文件路径
            metadata: 元数据（可选）
            
        Returns:
            bool: 添加是否成功
        """
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 添加文档元数据
            doc_metadata = {
                "source": file_path,
                "type": "file",
                "size": len(text),
                "lines": len(text.split('\n'))
            }
            if metadata:
                doc_metadata.update(metadata)
            
            return self.add_documents([text], [doc_metadata])
            
        except Exception as e:
            logger.error(f"从文件 {file_path} 添加文档失败: {e}")
            return False
    
    def query(self, 
               question: str, 
               chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        执行 RAG 查询
        
        Args:
            question: 用户问题
            chat_history: 对话历史（可选）
            
        Returns:
            str: 生成的回答
        """
        try:
            # 检索相关文档
            context_docs = self.retriever.invoke(question)
            system_message = f"""如果知识上下文中没有相关信息，请说明知识库中没有相关内容。   
                                要求：
                                    1. 回答要准确、简洁、有条理
                                    2. 如果有多条相关信息，请综合起来回答
                                    3. 不要编造知识上下文中没有的信息
                                    4. 保持友好的语调
                        """           
            # 获取对话历史
            if self.message_history is not None:
                # 使用新的消息历史系统
                messages = self.message_history.messages
                processed_messages = self.message_processor(messages)
            else:
                # 使用简单记忆或外部传入的历史
                processed_messages = []
                if chat_history:
                    processed_messages = self._format_chat_history(chat_history)
                elif hasattr(self, 'simple_memory'):
                    processed_messages = self._format_chat_history(self.simple_memory)

            # 执行 RAG 链
            response = self.rag_chain.invoke({
                "context": context_docs,
                "question": question,
                "system_message": system_message,
                "chat_history": processed_messages
            })
            
            # 保存到记忆系统
            if self.message_history is not None:
                self.message_history.add_user_message(question)
                self.message_history.add_ai_message(response)
            elif hasattr(self, 'simple_memory'):
                self.simple_memory.append(("user", question))
                self.simple_memory.append(("assistant", response))
            
            return response
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"抱歉，处理您的问题时出现了错误: {e}"
    
    def search_documents(self, 
                       query: str, 
                       k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询字符串
            k: 返回的文档数量
            
        Returns:
            List[Dict]: 相关文档信息
        """
        try:
            # 执行相似性搜索
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # 格式化结果
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "similarity": 1 - (score / 2)  # 将分数转换为相似度
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"搜索文档失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            Dict: 集合统计信息
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            # 获取样本文档大小分布
            avg_length = 0
            if count > 0:
                sample = collection.peek(limit=10)
                # 检查sample结构是否正确
                if sample and 'documents' in sample and sample['documents']:
                    documents = sample['documents']
                    avg_length = sum(len(doc) if isinstance(doc, str) else len(doc.get('document', '')) for doc in documents) / len(documents)

            # 获取嵌入模型名称，根据不同类型的嵌入模型使用不同的属性
            embedding_model_name = ""
            if hasattr(self.embedding_model, 'model'):
                embedding_model_name = self.embedding_model.model
            elif hasattr(self.embedding_model, 'model_name'):
                embedding_model_name = self.embedding_model.model_name
            elif hasattr(self.embedding_model, 'model'):
                embedding_model_name = self.embedding_model.model
            else:
                embedding_model_name = str(type(self.embedding_model).__name__)
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "avg_chunk_length": avg_length,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": embedding_model_name,
                "llm_model": self.llm.model_name
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            Dict: 系统状态信息
        """
        try:
            # 获取嵌入模型名称，根据不同类型的嵌入模型使用不同的属性
            embedding_model_name = ""
            if hasattr(self.embedding_model, 'model'):
                embedding_model_name = self.embedding_model.model
            elif hasattr(self.embedding_model, 'model_name'):
                embedding_model_name = self.embedding_model.model_name
            else:
                embedding_model_name = str(type(self.embedding_model).__name__)
            
            # 获取LLM提供商和模型名称
            llm_provider = os.getenv("LLM_TYPE", "openai")
            llm_model_name = ""
            
            if hasattr(self.llm, 'model'):
                llm_model_name = self.llm.model
            elif hasattr(self.llm, 'model_name'):
                llm_model_name = self.llm.model_name
            elif hasattr(self.llm, 'repo_id'):
                llm_model_name = self.llm.repo_id
            else:
                llm_model_name = str(type(self.llm).__name__)
            
            return {
                "llm_provider": llm_provider,
                "llm_model": llm_model_name,
                "embedding_model": embedding_model_name,
                "collection_name": self.collection_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "status": "ready"
            }
        except Exception as e:
            logger.error(f"获取状态信息失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        清空集合
        
        Returns:
            bool: 清空是否成功
        """
        try:
            # 删除并重新创建集合
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(self.collection_name)
            
            # 重新初始化向量存储
            self._initialize_collection()
            
            logger.info(f"集合 {self.collection_name} 已清空")
            return True
            
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False
    
    def clear_conversation(self) -> Dict[str, Any]:
        """
        清空对话历史
        
        Returns:
            Dict: 操作结果
        """
        try:
            if self.message_history is not None:
                self.message_history.clear()
            elif hasattr(self, 'simple_memory'):
                self.simple_memory.clear()
            
            return {
                "success": True,
                "message": "对话历史已清空"
            }
        except Exception as e:
            logger.error(f"清空对话历史失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        获取对话历史
        
        Returns:
            List: 对话历史记录
        """
        try:
            if self.message_history is not None:
                messages = self.message_history.messages
                # 转换为字典格式
                formatted_history = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        formatted_history.append({
                            "role": "user",
                            "content": msg.content,
                            "timestamp": getattr(msg, 'timestamp', None)
                        })
                    elif isinstance(msg, AIMessage):
                        formatted_history.append({
                            "role": "assistant",
                            "content": msg.content,
                            "timestamp": getattr(msg, 'timestamp', None)
                        })
                return formatted_history
            elif hasattr(self, 'simple_memory'):
                # 转换简单记忆为字典格式
                formatted_history = []
                for role, content in self.simple_memory:
                    formatted_history.append({
                        "role": role,
                        "content": content,
                        "timestamp": None
                    })
                return formatted_history
            else:
                return []
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        获取记忆系统信息
        
        Returns:
            Dict: 记忆系统信息
        """
        try:
            if self.message_history is not None:
                return {
                    "memory_type": self.memory_type,
                    "memory_class": "InMemoryChatMessageHistory",
                    "message_count": len(self.message_history.messages),
                    "window_size": self.memory_k if self.memory_type == "window" else None,
                    "summary": getattr(self, 'conversation_summary', None)
                }
            else:
                return {
                    "memory_type": "simple",
                    "memory_class": "SimpleMemory",
                    "conversation_length": len(self.simple_memory) if hasattr(self, 'simple_memory') else 0
                }
        except Exception as e:
            logger.error(f"获取记忆信息失败: {e}")
            return {"error": str(e)}
    
    def create_conversation_chain(self):
        """
        创建带有记忆的对话链
        
        Returns:
            RunnableWithMessageHistory: 带有记忆的对话链
        """
        try:
            # 创建对话提示模板
            conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是Gientech的知识库助手。根据给定的对话历史，回答用户的问题。"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{question}"),
            ])
            
            # 使用 RunnableWithMessageHistory 包装 LLM
            runnable_with_history = RunnableWithMessageHistory(
                self.llm,
                history=self.message_history
            )
            
            return runnable_with_history
            
        except Exception as e:
            logger.error(f"创建对话链失败: {e}")
            return None

# 示例用法
if __name__ == "__main__":
    # 初始化 RAG 系统（带记忆功能）
    rag = RAGSystem(
        collection_name="demo_rag",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=200,
        memory_type="buffer"  # 可以是 "buffer", "window", "summary"
    )
    # 示例文档
    sample_docs = [
        "Python 是一种高级编程语言，以其简洁易读的语法而闻名。",
        "Python 由 Guido van Rossum 于 1991 年创建。",
        "Python 广泛用于 Web 开发、数据科学、人工智能等领域。",
        "LangChain 是一个用于构建 LLM 应用程序的框架。",
        "LangChain 提供了模块化的组件来构建 RAG 系统和智能代理。",
    ]
    
    # 添加文档
    print("添加文档...")
    rag.add_documents(sample_docs)
    
    # 获取统计信息
    stats = rag.get_collection_stats()
    print(f"集合统计: {stats}")
    
    # 执行查询
    print("\n执行查询...")

    question = "什么是 Python？"
    answer = rag.query(question)
    print(f"问题: {question}")
    print(f"回答: {answer}")
    
    # 搜索相关文档
    print("\n搜索相关文档...")
    search_results = rag.search_documents("Python 的特点", k=2)
    for i, result in enumerate(search_results):
        print(f"结果 {i+1}:")
        print(f"  相似度: {result['similarity']:.2f}")
        print(f"  内容: {result['content'][:100]}...")
    
    # 带记忆的连续对话
    print("\n带记忆的连续对话...")
    
    # 第一次查询
    question1 = "什么是 LangChain？"
    answer1 = rag.query(question1)
    print(f"问题: {question1}")
    print(f"回答: {answer1}")
    
    # 第二次查询（测试记忆功能）
    question2 = "它有什么特点？"
    answer2 = rag.query(question2)
    print(f"问题: {question2}")
    print(f"回答: {answer2}")
    
    # 查看记忆信息
    print("\n记忆系统信息:")
    memory_info = rag.get_memory_info()
    print(f"记忆类型: {memory_info['memory_type']}")
    print(f"记忆类: {memory_info['memory_class']}")
    
    # 查看对话历史
    print("\n对话历史:")
    history = rag.get_conversation_history()
    for i, msg in enumerate(history):
        print(f"{i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    # 测试清空对话
    print("\n清空对话历史...")
    clear_result = rag.clear_conversation()
    print(f"清空结果: {clear_result}")
    
    # 再次查看对话历史
    print("\n清空后的对话历史:")
    history = rag.get_conversation_history()
    print(f"历史记录数: {len(history)}")
