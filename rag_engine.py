import os
import sys
import shutil
import nest_asyncio

# --- LlamaIndex 核心組件 ---
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage, 
    Settings, 
    PromptTemplate
)
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# --- Google Gemini 模型 ---
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- 資料解析器 ---
from llama_parse import LlamaParse

# --- 關鍵字檢索 ---
from llama_index.retrievers.bm25 import BM25Retriever

# ===========================================
#  🔑 在這裡設定你的 API KEY (請填入你的金鑰)
# ===========================================
# 如果你使用了 .env 檔案，請取消下面兩行的註解並安裝 python-dotenv
# from dotenv import load_dotenv
# load_dotenv()

# 或者直接在這裡填入 (請小心不要外流)
# os.environ["GOOGLE_API_KEY"] = "..." 
# os.environ["LLAMA_CLOUD_API_KEY"] = "..."
# ===========================================

nest_asyncio.apply()

# ================= 全域配置 =================

# 1. 資料夾配置
DATA_DIR = "./data"      
PERSIST_DIR = "./storage" 

# ===========================================

def init_settings():
    """初始化 LlamaIndex 全域設定"""
    
    # 這裡會直接抓取上面 os.environ 設定好的 Key
    Settings.llm = Gemini(model="models/gemini-2.5-flash")
    
    Settings.embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
        # 這裡其實不用特別傳 api_key，Gemini 會自動讀取環境變數，但保留也無妨
        api_key=os.environ.get("GOOGLE_API_KEY") 
    )
    
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 200
    Settings.embed_batch_size = 10 

def get_index(force_reload=False):
    """獲取向量索引"""
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 已建立資料目錄：{DATA_DIR}，請放入檔案。")

    if force_reload and os.path.exists(PERSIST_DIR):
        print(f"🧹 [System] 強制重建，清除舊索引目錄：{PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)

    if os.path.exists(PERSIST_DIR):
        print(f"📂 [Storage] 發現現有索引，直接載入...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index

    else:
        print("🚀 [ETL] 開始讀取 ./data 資料夾並建立索引...")
        
        # LlamaParse 會自動讀取 os.environ["LLAMA_CLOUD_API_KEY"]
        parser = LlamaParse(
            result_type="markdown",
            verbose=True,
            language="ch_tra",
            parsing_instruction="請將這份文件解析為標準 Markdown，保留表格結構與關鍵數據。"
        )
        
        file_extractor = {".pdf": parser}
        
        if not os.listdir(DATA_DIR):
            print("⚠️ 資料夾是空的，請先上傳檔案！")
            return None

        documents = SimpleDirectoryReader(
            input_dir=DATA_DIR,           
            file_extractor=file_extractor, 
            recursive=True                
        ).load_data()
        
        print(f"📄 共讀取了 {len(documents)} 個文件片段")

        print("⚡ [Vector Store] 正在建立 Vector Index...")
        index = VectorStoreIndex.from_documents(documents)
        
        print(f"💾 [Storage] 儲存索引至 {PERSIST_DIR}...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        
        return index

class CustomHybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle):
        try:
            vec_nodes = self.vector_retriever.retrieve(query_bundle)
            bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
            
            all_nodes = {}
            for node in vec_nodes: all_nodes[node.node.node_id] = node
            for node in bm25_nodes:
                if node.node.node_id not in all_nodes: all_nodes[node.node.node_id] = node
            
            return list(all_nodes.values())[:20]
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return []

def create_hybrid_query_engine(index):
    print("🔧 [Factory] 初始化混合檢索器...")
    
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
    retriever = CustomHybridRetriever(vector_retriever, bm25_retriever)

    qa_prompt_str = (
        "以下是參考文件內容：\n---------------------\n{context_str}\n---------------------\n"
        "請僅根據上述參考文件內容，回答使用者的問題: {query_str}\n"
        "嚴格禁止編造文件中未提及的人名、數字或職稱。\n"
        "請務必使用「繁體中文」回答。\n"
    )
    
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=PromptTemplate(qa_prompt_str)
    )
    
# ==========================================
#  測試用入口 (只有直接執行此檔案時才會跑)
# ==========================================
if __name__ == "__main__":
    print("🏁 開始獨立測試 rag_engine (使用硬編碼 API Key)...")
    
    try:
        # 1. 初始化設定 (會讀取最上面 os.environ 設定的 Key)
        print("⚙️  正在初始化設定...")
        init_settings()
        
        # 2. 執行建庫 (force_reload=True 代表強制重建)
        print("🚀 呼叫 get_index()...")
        index = get_index(force_reload=True)
        
        if index:
            print("✅ 測試成功！索引已建立並儲存至 ./storage")
        else:
            print("⚠️ 測試結束，但沒有建立索引 (可能是資料夾為空)")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        # 印出詳細錯誤以便除錯
        import traceback
        traceback.print_exc()