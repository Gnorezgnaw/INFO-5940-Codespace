import streamlit as st
import os
from os import environ
import streamlit as st

# LangChain / OpenAI client pieces
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF loader (pypdf)
from langchain.document_loaders import PyPDFLoader

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    st.warning("Please set OPENAI_API_KEY in the environment or Streamlit secrets.")

llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.2,
)


def load_documents_from_uploads(files):
    """Load a list of uploaded files into LangChain Document objects.

    Supports .txt, .md (read as text) and .pdf via PyPDFLoader.
    """
    docs = []
    for uploaded in files:
        name = uploaded.name
        data = uploaded.read()
        if name.lower().endswith(".pdf"):
            # PyPDFLoader expects a file on disk; write to a temp file
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            loaded = loader.load()
            # add source metadata
            for d in loaded:
                d.metadata = d.metadata or {}
                d.metadata["source"] = name
            docs.extend(loaded)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # treat as text
            text = data.decode("utf-8")
            docs.append(Document(page_content=text, metadata={"source": name}))
    return docs


def build_vectorstore(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="openai.text-embedding-3-large")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore


st.title("📝 File Q&A (RAG) with OpenAI")

uploaded_files = st.file_uploader("Upload document(s)", type=("txt", "md", "pdf"), accept_multiple_files=True)

# If new files were uploaded this run, load them and store in session_state to avoid re-reading
if uploaded_files:
    if "raw_docs" not in st.session_state:
        st.session_state["raw_docs"] = {}
    # Load only files whose source is not already in session_state (avoid duplicates)
    loaded = load_documents_from_uploads(uploaded_files)
    for d in loaded:
        src = d.metadata.get("source", "(no source)")
        st.session_state["raw_docs"].setdefault(src, [])
        # avoid adding exact duplicates (basic dedupe by text beginning)
        existing = st.session_state["raw_docs"][src]
        if not any(e.page_content[:64] == d.page_content[:64] for e in existing):
            st.session_state["raw_docs"][src].append(d)

# Show currently loaded documents and allow the user to select which to include in the index
available_sources = list(st.session_state.get("raw_docs", {}).keys())
selected_sources = available_sources
if available_sources:
    selected_sources = st.multiselect("Select documents to include in the knowledge base", available_sources, default=available_sources)

question = st.chat_input("Ask something about the uploaded documents", disabled=not available_sources)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Upload one or more documents and ask questions about them."}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


PROMPT_TMPL = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer from the context, say you don't know. Be concise (<= 3 sentences).

Question: {question}

Context: {context}

Answer:
"""


if question and uploaded_files:
    # load and index
    # Build list of documents to index based on selected_sources
    docs_to_index = []
    for src in selected_sources:
        docs_to_index.extend(st.session_state.get("raw_docs", {}).get(src, []))

    if not docs_to_index:
        st.error("No documents selected for indexing. Choose at least one document.")
        st.stop()

    with st.spinner("Building index for selected documents — this may take a moment..."):
        # Rebuild the vectorstore only when the selection changes
        last_sel = st.session_state.get("last_selected_sources")
        if last_sel != tuple(selected_sources) or "vectorstore" not in st.session_state:
            vectorstore = build_vectorstore(docs_to_index)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["last_selected_sources"] = tuple(selected_sources)
        else:
            vectorstore = st.session_state["vectorstore"]

        # retrieve — use the vectorstore API directly for compatibility across langchain versions
        # some Retriever objects do not expose `get_relevant_documents`; use similarity_search instead.
        retrieved = vectorstore.similarity_search(question, k=5)

    # format context text
    context_text = "\n\n---\n\n".join(d.page_content for d in retrieved)

    system_instructions = PROMPT_TMPL.format(question=question, context=context_text)

    # Compose messages and call the LLM
    messages = [SystemMessage(content=system_instructions), HumanMessage(content=question)]
    result = llm.invoke(messages)
    answer = getattr(result, "content", str(result))

    # show the user interaction
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)