# 長庚大學 碩士論文與畢業離校流程 RAG 助手 (CGU-Grad-RAG)

這是一個基於 RAG (Retrieval-Augmented Generation) 技術的 AI 助手，專門用於回答關於長庚大學碩士論文格式規範與畢業離校流程的相關問題。

## ⚠️ 前置準備與 API 金鑰

本專案使用 Google Gemini 模型與 LlamaCloud 解析器，您需要準備以下 API Key：

1.  **Google Gemini API Key**: 用於 LLM 生成與 Embedding。
2.  **LlamaCloud API Key**: 用於高品質解析 PDF 文件 (LlamaParse)。

### 設定 LlamaCloud API Key
由於介面上主要提供 Google API Key 的輸入，建議您將 `LLAMA_CLOUD_API_KEY` 設定於環境變數中，或建立 `.env` 檔案（如果不打算上傳至 GitHub）：

1. 在專案根目錄建立 `.env` 檔案。
2. 加入以下內容：
    ```env
    LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
    ```
   *(程式碼中有支援 `python-dotenv`，若您已安裝並取消註解相關程式碼即可自動讀取)*

## 🛠️ 環境安裝

1.  **安裝 Python**: 建議使用 Python 3.8 或以上版本。
2.  **Clone 專案**:
    ```bash
    git clone https://github.com/BroJack0809/-RAG-.git
    cd -RAG-
    ```
    *(註：請根據實際 Clone 下來的資料夾名稱進入目錄)*

3.  **安裝相依套件**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 如何啟動專案

在終端機 (Terminal) 或命令提示字遠 (CMD) 中執行以下指令：

```bash
python -m streamlit run app.py
```

啟動後，瀏覽器應會自動打開 `http://localhost:8501`。

## 📖 如何使用

1.  **輸入 API Key**:
    - 在左側側邊欄的「系統設定」中，輸入您的 **Google API Key**。
    
2.  **上傳文件**:
    - 在「知識庫管理」區域，點擊「Browse files」上傳您的 PDF 或 Word 文件 (例如：離校手續說明、論文格式規範等)。
    - 上傳後，檔案會被儲存至 `./data` 資料夾。

3.  **建立/重建 索引**:
    - 點擊側邊欄的 **「🔄 重建知識庫 (Re-Index)」** 按鈕。
    - 系統會開始解析文件 (使用 LlamaParse) 並建立向量索引。這可能需要幾分鐘的時間，請耐心等待。
    - 完成後會顯示「✅ 知識庫重建完成！」。

4.  **開始對話**:
    - 在主畫面的對話框中輸入您的問題，例如：「畢業離校流程有哪些步驟？」或「論文邊界要設定多少？」。
    - AI 將根據您上傳的文件內容回答問題，並附上參考來源與相關片段。

## 📁 專案結構

- `app.py`: Streamlit 前端應用程式主檔。
- `rag_engine.py`: RAG 核心邏輯 (LlamaIndex, Gemini, LlamaParse 設定)。
- `data/`: 存放使用者上傳的文件。
- `storage/`: 存放建立好的向量索引 (Vector Store)。
- `requirements.txt`: 專案相依套件列表。

## 📝 注意事項

- **PDF 解析**: 若未設定 `LLAMA_CLOUD_API_KEY`，PDF 解析功能可能會失效或退回預設解析器，建議務必申請並設定。
- **Google API 限制**: 若遇到 `429` 錯誤，表示達到 Google API 使用頻率上限，請稍待片刻再試。
