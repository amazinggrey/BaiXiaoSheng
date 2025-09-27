# 百晓生知识库检索系统

基于LangChain框架的智能知识检索平台，支持多种文件格式的上传和文本输入，通过本地或云端模型实现高效的向量检索和对话功能。

## 功能特性

- 📄 **多格式文件支持**: 支持TXT、PDF、DOCX、PPTX、XLSX、XLS、CSV文件上传
- 🔤 **文本输入**: 支持直接输入文本内容添加到知识库
- 🧠 **智能分块**: 自动将长文本分割成合适的片段
- 🔍 **向量检索**: 支持本地OLLAMA模型、sentence-transformers和OpenAI嵌入
- 💬 **RAG对话**: 结合知识库的智能对话功能，支持上下文记忆
- 🌐 **友好界面**: 响应式Web界面，支持拖拽上传
- ⚡ **实时搜索**: 快速检索相关知识内容
- 🎯 **语义搜索**: 基于向量嵌入的语义相似度搜索
- 🔧 **灵活配置**: 支持多种嵌入模型和LLM提供商

## 系统架构

### LangChain RAG架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   Flask后端      │    │  LangChain核心   │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ 智能对话    │ │    │ │ RAG系统     │ │    │ │ 嵌入模型    │ │
│ │ 文件上传    │ │    │ │ 文档处理    │ │    │ │ 向量存储    │ │
│ │ 文本输入    │ │◄──►│ │ API接口     │ │◄──►│ │ 检索链      │ │
│ │ 知识检索    │ │    │ │ 对话管理    │ │    │ │ 对话链      │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘           │
                                │                   │
                                ▼                   ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  嵌入模型      │    │  LLM提供商     │
                       │                 │    │                 │
                       │ ┌─────────────┐ │    │ ┌─────────────┐ │
                       │ │ Ollama     │ │    │ │ OpenAI      │ │
                       │ │ HuggingFace│ │    │ │ Ollama      │ │
                       │ │ OpenAI     │ │    │ │ 其他LLM     │ │
                       │ └─────────────┘ │    │ └─────────────┘ │
                       └─────────────────┘    └─────────────────┘
```

### 核心组件

1. **langchain_embedding_utils.py**: 嵌入模型管理
   - 支持Ollama、HuggingFace、OpenAI嵌入模型
   - 文本分块和批量处理
   - 向量生成和管理

2. **langchain_vector_store.py**: 向量存储管理
   - 基于ChromaDB的持久化存储
   - 多格式文档加载和处理
   - 相似度搜索和评分

3. **langchain_rag_system.py**: RAG系统核心
   - QA链和对话链管理
   - 对话历史记忆
   - 多LLM提供商支持

4. **app_langchain.py**: Flask应用接口
   - RESTful API设计
   - 文件上传和处理
   - 实时对话和搜索

### RAG处理流程

1. **文档摄入**: 支持多种格式文件上传和文本输入
2. **文档处理**: 自动分块、元数据提取、向量嵌入
3. **向量存储**: 持久化到ChromaDB向量数据库
4. **用户查询**: 语义搜索和相似度匹配
5. **上下文构建**: 组合检索到的相关文档
6. **RAG生成**: 结合上下文通过LLM生成回答
7. **对话管理**: 保存历史记录，支持上下文对话

## 快速开始

### 1. 环境要求

- Python 3.8+
- Ollama (可选，用于本地模型)
- OpenAI API Key (可选，用于云端模型)

### 2. 安装步骤

1. **克隆或下载项目**
   ```bash
   # 如果您有git仓库
   git clone <repository-url>
   cd raggientech
   
   # 或者直接下载解压到目录
   ```

2. **运行启动脚本**
   ```bash
   python app_langchain.py
   ```

   系统会自动：
   - 初始化LangChain组件
   - 配置嵌入模型和向量存储
   - 启动Flask应用
   - 打开浏览器界面

### 3. 配置环境变量

在 `.env` 文件中配置模型参数：

```env
# 嵌入模型配置
EMBEDDING_MODEL_TYPE=ollama
EMBEDDING_MODEL_NAME=nomic-embed-text:v1.5
OLLAMA_BASE_URL=http://localhost:11434

# LLM配置
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_openai_api_key_here

# 文本处理配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 向量存储配置
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=knowledge_base
```

### 4. 访问应用

浏览器会自动打开 `http://localhost:5000`

## 使用说明

### 添加知识

1. **文件上传**
   - 支持格式：TXT、PDF、DOCX、PPTX、XLSX、XLS、CSV
   - 拖拽或点击上传文件
   - 自动解析和处理文档内容

2. **文本输入**
   - 直接输入文本内容
   - 可添加标题和元数据
   - 实时添加到知识库

### 智能对话

1. **RAG对话**
   - 基于知识库内容回答问题
   - 支持上下文连续对话
   - 提供来源文档引用

2. **对话管理**
   - 查看对话历史
   - 清空对话记录
   - 导出对话内容

### 知识检索

1. **语义搜索**
   - 支持自然语言查询
   - 返回相关文档片段
   - 显示相似度评分

2. **结果过滤**
   - 按文件类型筛选
   - 按来源分类显示
   - 自定义返回数量

## API文档

### 系统信息

```http
GET /api/system_info
```

### 文件上传

```http
POST /api/upload
Content-Type: multipart/form-data

file: [文件对象]
```

### 文本添加

```http
POST /api/add_text
Content-Type: application/json

{
  "text": "文本内容",
  "title": "标题（可选）"
}
```

### 知识搜索

```http
POST /api/search
Content-Type: application/json

{
  "query": "搜索查询",
  "k": 5
}
```

### RAG对话

```http
POST /api/chat
Content-Type: application/json

{
  "query": "用户问题",
  "use_conversation": true
}
```

### 健康检查

```http
GET /api/health
```

### 对话管理

```http
GET /api/conversation_history
POST /api/clear_conversation
```

### 知识库管理

```http
POST /api/clear_knowledge
GET /api/export
GET /api/file_types
```

## 配置说明

### 环境变量配置

```env
# 嵌入模型配置
EMBEDDING_MODEL_TYPE=ollama|sentence-transformers|openai
EMBEDDING_MODEL_NAME=nomic-embed-text:v1.5|all-MiniLM-L6-v2|text-embedding-ada-002
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_DIMENSION=768

# LLM配置
LLM_PROVIDER=openai|ollama
LLM_MODEL=gpt-3.5-turbo|gpt-4|llama2|mistral
OPENAI_API_KEY=your_api_key_here

# 文本处理配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 向量存储配置
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# 对话配置
MAX_CONVERSATION_LENGTH=10
```

### 模型选择指南

1. **本地部署推荐**:
   - 嵌入模型: `nomic-embed-text:v1.5` (Ollama)
   - LLM: `llama2` 或 `mistral` (Ollama)

2. **云端部署推荐**:
   - 嵌入模型: `text-embedding-ada-002` (OpenAI)
   - LLM: `gpt-3.5-turbo` 或 `gpt-4` (OpenAI)

3. **混合部署**:
   - 本地嵌入 + 云端LLM
   - 云端嵌入 + 本地LLM

## 目录结构

```
raggientech/
├── app_langchain.py              # LangChain Flask应用
├── langchain_vector_store.py     # 向量存储管理
├── langchain_embedding_utils.py  # 嵌入模型工具
├── langchain_rag_system.py       # RAG系统核心
├── test_langchain.py            # 测试脚本
├── requirements.txt             # Python依赖
├── .env                        # 环境变量配置
├── README.md                   # 项目说明文档
├── templates/
│   └── index_rag.html          # 前端界面
├── chroma_db/                  # ChromaDB数据目录
└── uploads/                    # 临时文件目录（可选）
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查Ollama服务是否运行: `ollama ps`
   - 确认模型已下载: `ollama pull nomic-embed-text:v1.5`
   - 检查API密钥配置

2. **向量存储错误**
   - 检查ChromaDB目录权限
   - 确认嵌入模型正常工作
   - 清空数据库重新初始化

3. **文件处理失败**
   - 检查文件格式支持
   - 确认文件编码(UTF-8)
   - 查看文件大小限制

4. **对话功能异常**
   - 检查LLM配置
   - 确认API调用限额
   - 查看网络连接状态

### 性能优化

1. **文本分块优化**
   - 根据内容类型调整CHUNK_SIZE
   - 合理设置重叠度提高连续性

2. **搜索优化**
   - 调整返回结果数量
   - 使用更具体的查询词汇

3. **模型优化**
   - 选择适合硬件的模型
   - 批量处理提高效率

## 开发说明

### 扩展功能

1. **支持更多文件格式**
   - 在langchain_vector_store.py中添加加载器
   - 安装相应依赖包

2. **集成新的嵌入模型**
   - 在langchain_embedding_utils.py中添加支持
   - 配置模型参数

3. **添加新的LLM提供商**
   - 在langchain_rag_system.py中集成
   - 实现相应的接口

### 自定义配置

1. **修改默认参数**
   - 调整.env文件中的配置
   - 重启应用使配置生效

2. **更换向量数据库**
   - 替换ChromaDB为其他向量存储
   - 修改相关接口调用

## 版本信息

- LangChain: 0.3.27
- langchain-openai: 0.3.32
- langchain-community: 0.3.29
- langchain-chroma: 0.2.6
- ChromaDB: 1.1.0

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用本系统需要相应的API密钥或本地模型。请确保遵守相关服务提供商的使用条款和隐私政策。
