# Chat with PDF

This Streamlit app allows you to upload text, markdown, or PDF documents and ask natural language questions about their contents.  
The application integrates **OpenAI models** (via Cornell's AI Gateway) with **LangChain** and **Chroma** to enable Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- **File Upload**: Upload `.txt`, `.pdf`, or `.md` documents directly in the Streamlit interface.
- **Conversational Q&A**: Ask questions about the uploaded document(s) and get streamed answers from OpenAI's GPT-4o.
- **Context-Aware Retrieval**: Uses LangChain’s `RecursiveCharacterTextSplitter` and Chroma vector store for semantic chunking and retrieval.
- **Inline Citations**: Answers can include citations like `[source - chunk N]` pointing back to relevant parts of the document.
- **Session Memory**: Maintains chat history for a smoother conversational experience.
- **Transparency**: Expander view shows the retrieved document chunks for debugging and validation.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo>
```
---

## Running a Streamlit App on Codespaces  
Follow these steps to launch and view your Streamlit app in GitHub Codespaces:
1. **Open the terminal** inside your Codespace.
2. Run the command:  
   ```bash
   streamlit run your-file-name.py
   ```  
   **(Replace `your-file-name.py` with the actual name of your Streamlit app file, e.g., `hello_app.py`.)**
3. After pressing **Enter**, a popup should appear in the bottom-right corner of Codespace editor.  
   - Click **“Open in Browser”** to view your app.  

   ⚠️ *If you miss the popup:*  
   - Press **Ctrl + C** in the terminal to stop the app.  
   - Rerun the command from step 2 — the popup should appear again.
4. A new browser tab will open, showing the interface of your Streamlit app.
5. **Make changes to your code** in the Codespace editor.  
   - Refresh the browser tab to see the updated version of your app.  

## Troubleshooting
- The Jupyter extension should install automatically. If you still cannot select a Python kernel on Jupyter Notebook: Go to the left sidebar >> **Extensions** >> search for **Jupyter** >> reload window (or reinstall it).   
