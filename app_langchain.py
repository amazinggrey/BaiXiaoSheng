#!/usr/bin/env python3
"""
基于 LangChain 框架的 RAG 知识库系统
支持多种文档格式，包括 Excel
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
import json
from dotenv import load_dotenv
from langchain_vector_store import langchain_vector_store
from langchain_rag_system import RAGSystem as langchain_rag_system
from langchain_embedding_utils import langchain_embedding_utils

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)

# 初始化 RAG 系统
rag_system = langchain_rag_system()

def _safe_delete_file(file_path):
    """安全删除文件，处理文件锁定问题"""
    import time
    
    max_retries = 5
    retry_delay = 0.5  # 秒
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                print(f"✓ 临时文件已删除: {file_path}")
                return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"⚠ 文件被锁定，等待 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print(f"✗ 无法删除临时文件: {file_path}, 错误: {e}")
                return False
        except Exception as e:
            print(f"✗ 删除临时文件时发生错误: {file_path}, 错误: {e}")
            return False
    
    return False

def allowed_file(filename):
    """检查文件类型是否允许"""
    allowed_extensions = {'txt', 'pdf', 'docx', 'pptx', 'xlsx', 'xls', 'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """主页"""
    return render_template('index_rag.html')

@app.route('/api/system_info', methods=['GET'])
def get_system_info():
    """获取系统信息"""
    try:
        return jsonify({
            "embedding_info": langchain_embedding_utils.get_model_info(),
            "vectorstore_info": langchain_vector_store.get_status(),
            "rag_info": rag_system.get_status()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """文件上传和处理"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "没有文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "不支持的文件类型"}), 400
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            file.save(tmp_file.name)
            
            try:
                # 加载文档
                metadata = {
                    "upload_time": json.dumps({"timestamp": "now"}),  # 简化时间戳
                    "original_filename": file.filename
                }
                
                documents = langchain_vector_store.load_document(tmp_file.name, metadata)
                
                if not documents:
                    return jsonify({"error": "文档加载失败"}), 500
                
                # 添加到向量存储
                success = langchain_vector_store.add_documents(documents)
                
                if success:
                    return jsonify({
                        "message": "文件上传成功",
                        "filename": file.filename,
                        "document_count": len(documents),
                        "total_count": langchain_vector_store.get_document_count()
                    }), 200
                else:
                    return jsonify({"error": "向量存储添加失败"}), 500
                    
            finally:
                # 安全删除临时文件，处理文件锁定问题
                _safe_delete_file(tmp_file.name)
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_text', methods=['POST'])
def add_text():
    """添加文本到知识库"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        title = data.get('title', '')
        
        if not text.strip():
            return jsonify({"error": "文本内容为空"}), 400
        
        # 创建文档
        from langchain.schema import Document
        
        content = f"{title}\n\n{text}" if title else text
        
        # 添加元数据
        metadata = {
            "source": title or "manual_input",
            "type": "text",
            "title": title,
            "created_time": json.dumps({"timestamp": "now"})  # 简化时间戳
        }
        
        document = Document(page_content=content, metadata=metadata)
        
        # 添加到向量存储
        success = langchain_vector_store.add_documents([document])
        
        if success:
            return jsonify({
                "message": "文本添加成功",
                "document_count": 1,
                "total_count": langchain_vector_store.get_document_count()
            }), 200
        else:
            return jsonify({"error": "文本添加失败"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """知识搜索"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)
        
        if not query.strip():
            return jsonify({"error": "查询内容为空"}), 400
        
        # 执行搜索
        results = langchain_vector_store.similarity_search_with_score(query, k)
        
        return jsonify({
            "results": results,
            "count": len(results)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """RAG 聊天"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        use_conversation = data.get('use_conversation', True)
        
        if not query.strip():
            return jsonify({"error": "查询内容为空"}), 400
        
        # 执行 RAG 查询
        chat_history = [] if not use_conversation else None
        result = rag_system.query(query, chat_history)
        
        return jsonify({
            "success": True,
            "response": result,
            "query": query
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        return jsonify({
            "status": "healthy",
            "embedding_status": langchain_embedding_utils.get_model_info(),
            "vectorstore_status": langchain_vector_store.get_status(),
            "rag_status": rag_system.get_status()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_knowledge', methods=['POST'])
def clear_knowledge():
    """清空知识库"""
    try:
        success = langchain_vector_store.clear_collection()
        if success:
            return jsonify({
                "message": "知识库已清空",
                "count": 0
            }), 200
        else:
            return jsonify({"error": "清空失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_conversation', methods=['POST'])
def clear_conversation():
    """清空对话历史"""
    try:
        result = rag_system.clear_conversation()
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation_history', methods=['GET'])
def get_conversation_history():
    """获取对话历史"""
    try:
        history = rag_system.get_conversation_history()
        return jsonify({
            "conversation_history": history,
            "length": len(history)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export', methods=['GET'])
def export_knowledge():
    """导出知识库信息"""
    try:
        return jsonify({
            "status": langchain_vector_store.get_status(),
            "embedding_info": langchain_embedding_utils.get_model_info(),
            "rag_info": rag_system.get_status()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/file_types', methods=['GET'])
def get_supported_file_types():
    """获取支持的文件类型"""
    return jsonify({
        "supported_types": [
            {"extension": "txt", "description": "文本文件"},
            {"extension": "pdf", "description": "PDF 文档"},
            {"extension": "docx", "description": "Word 文档"},
            {"extension": "pptx", "description": "PowerPoint 演示文稿"},
            {"extension": "xlsx", "description": "Excel 工作簿"},
            {"extension": "xls", "description": "Excel 工作簿（旧版）"},
            {"extension": "csv", "description": "CSV 文件"}
        ]
    }), 200

if __name__ == '__main__':
    print("=== 基于 LangChain 的 RAG 知识库系统 ===")
    print("系统信息:")
    print(f"嵌入模型: {langchain_embedding_utils.get_model_info()}")
    print(f"向量存储: {langchain_vector_store.get_status()['vectorstore_status']}")
    print(f"RAG 系统: {rag_system.get_status()['llm_provider']} - {rag_system.get_status()['llm_model']}")
    print("=" * 50)
    print("访问地址: http://localhost:5000")
    print("支持的文件类型: txt, pdf, docx, pptx, xlsx, xls, csv")
    print("按 Ctrl+C 停止服务器")
    
    # 启动 Flask 应用
    app.run(debug=True, host='0.0.0.0', port=5000)
