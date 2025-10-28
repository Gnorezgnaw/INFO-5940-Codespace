# Reference Log (ref-log.md)

## External Sources and Tools Used

- **OpenAI API (Cornell Gateway)**  
  Used to run GPT-4o for answering questions and `text-embedding-3-large` for building embeddings.  
  Endpoint: `https://api.ai.it.cornell.edu`.

- **LangChain**  
  Framework used for:
  - Document loading (`TextLoader`, `PyPDFLoader`).
  - Text chunking (`RecursiveCharacterTextSplitter`).
  - Prompt construction (`PromptTemplate`).
  - Integrating retriever with the LLM.

- **Chroma**  
  Vector store used for semantic search and retrieval-augmented generation (RAG).

- **pypdf / PyPDFLoader**  
  For parsing uploaded PDF files and extracting their content into LangChain documents.

- **Streamlit**  
  Used as the web application framework to provide interactive UI for file uploads, chat input, and streamed responses.

---

## GenAI Usage

- **GitHub Copilot**  
  - Helped debug errors during development.  
  - Suggested UI refinements for the Streamlit chat interface.  
  - Assisted with boilerplate around session state handling.

- **ChatGPT**  
  - Helped draft this documentation (`README.md` and `ref-log.md`).  
  - Provided sample code patterns for RAG and streaming.  

---

## Notes

- GenAI contributions were limited to **debugging, scaffolding, and documentation writing**.  
- All substantive design decisions, configuration, and validation of correctness were made by the developer.  