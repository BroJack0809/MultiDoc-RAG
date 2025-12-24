#啟動app的指令: python -m streamlit run app.py
#CGU-Grad-RAG：學位論文格式與離校流程助手
# -*- coding: utf-8 -*-
import streamlit as st
import os
import nest_asyncio
import shutil
from llama_index.core import Settings

# 引入後端邏輯
# 注意：這裡假設您的後端檔案名為 rag_engine.py
import rag_engine as rag

nest_asyncio.apply()

# --- 頁面設定 ---
st.set_page_config(page_title="MultiDoc-RAG", layout="wide")
st.title("📚 MultiDoc-RAG")

# --- 側邊欄：設定與檔案管理 ---
with st.sidebar:
    # 1. 版本資訊區塊 (修正格式跑掉的問題)
    st.subheader("ℹ️ 資料來源依據日期")
    
    st.info(
        "**1. 離校手續流程**\n\n"
        "📅 文件日期：2025年05月20日\n\n"
        "---\n\n"
        "**2. 論文格式規範**\n\n"
        "📅 文件日期：2025年09月05日"
    )
    # 這裡使用 markdown 的分隔線讓視覺更乾淨
    st.markdown("---") 

    st.header("⚙️ 系統設定")
    
    # 2. API Key 設定
    default_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key = st.text_input("Google API Key", value=default_key, type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    # 2. 檔案上傳區
    st.header("📂 知識庫管理")
    uploaded_files = st.file_uploader(
        "上傳文件 (支援 PDF, Word)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # 確保 data 資料夾存在
        if not os.path.exists("./data"):
            os.makedirs("./data")
            
        # 儲存檔案
        for uploaded_file in uploaded_files:
            file_path = os.path.join("./data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"已上傳 {len(uploaded_files)} 個檔案至 ./data")

    # 3. 重建索引按鈕
    st.divider()
    if st.button("🔄 重建知識庫 (Re-Index)"):
        with st.spinner("正在重新解析文件並建立索引，這可能需要幾分鐘..."):
            # 清除快取，強制重跑
            st.cache_resource.clear()
            # 呼叫後端強制重建
            rag.init_settings()
            rag.get_index(force_reload=True)
            st.success("✅ 知識庫重建完成！")
            st.rerun()

# --- 檢查 API Key ---
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⬅️ 請先於側邊欄設定 Google API Key。")
    st.stop()

# --- 初始化引擎 ---
@st.cache_resource
def load_engine():
    try:
        rag.init_settings()
        # 預設不強制重建，只讀取現有的
        index = rag.get_index(force_reload=False)
        if index is None:
            return None
        return rag.create_hybrid_query_engine(index)
    except Exception as e:
        st.error(f"引擎初始化失敗: {e}")
        return None

# --- 主聊天介面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

engine = load_engine()

if engine is None:
    st.info("👋 歡迎使用！目前知識庫是空的。")
    st.warning("請在左側側邊欄上傳 PDF 或 Word 檔案，然後點擊「重建知識庫」按鈕來開始。")
else:
    if prompt := st.chat_input("請輸入問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI 正在檢索多份文件..."):
                try:
                    response = engine.query(prompt)
                    st.markdown(response.response)
                    
                    with st.expander("🕵️ 參考來源片段"):
                        for node in response.source_nodes:
                            # 顯示檔名 (Metadata) 讓你知道答案來自哪個檔案
                            file_name = node.node.metadata.get('file_name', '未知檔案')
                            score = f"{node.score:.2f}" if node.score is not None else "Hybrid"
                            st.caption(f"**[{file_name}] 分數: {score}**")
                            st.text(node.node.get_text()[:200] + "...")
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                except Exception as e:
                    if "429" in str(e):
                        st.error("⚠️ Google API 速度限制 (429)。請稍等幾分鐘後再試。")
                    else:
                        st.error(f"錯誤: {e}")