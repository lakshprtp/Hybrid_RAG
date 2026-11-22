import os
import streamlit as st
from dotenv import load_dotenv

import chat_history
import backend
from backend import (
    get_system_status,
    list_documents,
    clear_all_documents,
    index_files,
    execute_rag_query,
    QueryRequest
)

# Load environment
load_dotenv()

# Page setup
st.set_page_config(
    page_title="RAG AI Assistant | Gemini + Qdrant + Tavily",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4285F4, #34A853, #FBBC05, #EA4335);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-yellow { background-color: #fef9c3; color: #854d0e; }
    .badge-red { background-color: #fee2e2; color: #b91c1c; }
    .badge-blue { background-color: #e0f2fe; color: #0369a1; }
    .source-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .source-header {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .stButton > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Thread Management Setup
# ==========================================

# Ensure we have an active thread
threads = chat_history.list_threads()
if "current_thread_id" not in st.session_state or not st.session_state.current_thread_id:
    if threads:
        st.session_state.current_thread_id = threads[0]["id"]
    else:
        st.session_state.current_thread_id = chat_history.create_thread("New Chat")
        threads = chat_history.list_threads()


# ==========================================
# Sidebar Configuration
# ==========================================

with st.sidebar:
    # 1. New Thread Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = chat_history.create_thread("New Chat")
        st.session_state.current_thread_id = new_id
        st.rerun()

    # 2. Previous Chat Threads List
    st.markdown("### 💬 Chat History")
    threads = chat_history.list_threads()
    
    if threads:
        for t in threads:
            is_active = t["id"] == st.session_state.current_thread_id
            col_t, col_d = st.columns([0.82, 0.18])
            
            with col_t:
                btn_label = f"💬 {t['title'][:22]}..." if len(t['title']) > 22 else f"💬 {t['title']}"
                if is_active:
                    st.markdown(f"**👉 {btn_label}**")
                else:
                    if st.button(btn_label, key=f"thread_{t['id']}", use_container_width=True):
                        st.session_state.current_thread_id = t["id"]
                        st.rerun()
            
            with col_d:
                if st.button("🗑️", key=f"del_{t['id']}", help="Delete chat, Postgres messages, and Qdrant documents"):
                    del_thread_id = t["id"]
                    try:
                        # 1. Purge all document vectors from Qdrant Vector DB
                        clear_all_documents(thread_id=del_thread_id)
                    except Exception:
                        pass
                    try:
                        # 2. Purge thread and all conversation messages from PostgreSQL
                        chat_history.delete_thread(del_thread_id, purge_vector_db=False)
                    except Exception:
                        pass

                    # If deleted active thread, switch to another or create new
                    if del_thread_id == st.session_state.current_thread_id:
                        remaining = chat_history.list_threads()
                        st.session_state.current_thread_id = remaining[0]["id"] if remaining else chat_history.create_thread("New Chat")
                    st.rerun()
    else:
        st.caption("No saved conversations yet.")

    st.markdown("---")

    # 3. Search Strategy
    st.markdown("### 🔍 Search Strategy")
    search_mode = st.radio(
        "Choose retrieval strategy:",
        options=["hybrid", "documents", "web"],
        format_func=lambda x: {
            "hybrid": "⚡ Hybrid (Qdrant DB + Tavily Web)",
            "documents": "📚 Documents Only (Qdrant Vector DB)",
            "web": "🌐 Web Search Only (Tavily Search)"
        }[x],
        index=0
    )

    st.markdown("---")

    # 4. Document Ingestion
    st.markdown("### 📁 Document Ingestion")
    st.caption("Max file size: **2MB** per file (PDF, TXT, MD)")
    uploaded_files = st.file_uploader(
        "Upload files to Qdrant Knowledge Base",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    pdf_password = st.text_input(
        "PDF Password (if file is encrypted, e.g. e-Aadhaar)",
        type="password",
        help="For e-Aadhaar: First 4 letters of your name in CAPITAL + 4-digit Year of Birth (e.g. LAKS2001)"
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", min_value=200, max_value=2000, value=400, step=100)
    with col2:
        chunk_overlap = st.number_input("Overlap", min_value=0, max_value=500, value=100, step=50)

    if st.button("🚀 Index to Qdrant", use_container_width=True, disabled=not uploaded_files):
        MAX_UPLOAD_LIMIT_BYTES = 2 * 1024 * 1024
        oversized = [f for f in uploaded_files if len(f.getvalue()) > MAX_UPLOAD_LIMIT_BYTES]
        if oversized:
            for f in oversized:
                size_mb = round(len(f.getvalue()) / (1024 * 1024), 2)
                st.error(f"❌ File '{f.name}' ({size_mb}MB) exceeds the 2MB limit. Please upload files under 2MB.")
        else:
            with st.spinner("Chunking, computing Gemini embeddings, and indexing to Qdrant..."):
                try:
                    files_payload = [(f.name, f.getvalue()) for f in uploaded_files]
                    result = index_files(
                        files_data=files_payload,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        thread_id=st.session_state.current_thread_id,
                        pdf_password=pdf_password or None
                    )
                    st.success(f"Indexed {result.get('total_chunks')} chunks across {len(result.get('processed_files', []))} file(s) for this chat!")
                    st.rerun()
                except Exception as e:
                    err_detail = getattr(e, "detail", str(e))
                    st.error(f"Upload failed: {err_detail}")

    # Indexed Documents List (isolated to this chat thread)
    docs_info = list_documents(thread_id=st.session_state.current_thread_id)
    with st.expander(f"📚 Documents in this Chat ({docs_info.get('total_documents', 0)})"):
        if docs_info.get("documents"):
            for d in docs_info["documents"]:
                st.markdown(f"• **{d['filename']}** ({d['chunks']} chunks)")
            
            if st.button("🗑️ Clear Documents in this Chat", type="primary", use_container_width=True):
                try:
                    clear_all_documents(thread_id=st.session_state.current_thread_id)
                    st.success("Documents for this chat cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear: {e}")
        else:
            st.caption("No documents in this chat thread yet. Upload one above, or switch to a previous chat in the sidebar to access its documents.")

    st.markdown("---")

    # 5. Advanced Settings (Retrieval Settings & System Status)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        # Retrieval Settings
        st.markdown("#### 🎛️ Retrieval Settings")
        top_k_docs = st.slider("Document Chunks (Top-K)", min_value=1, max_value=20, value=8)
        max_web_results = st.slider("Web Results (Tavily)", min_value=1, max_value=10, value=4)
        tavily_depth = st.selectbox("Tavily Depth", ["basic", "advanced"], index=0)

        st.markdown("---")

        # System Status
        st.markdown("#### ⚙️ System Status")
        status_data = get_system_status()
        gemini_ok = status_data.get('gemini_api_configured', False)
        tavily_ok = status_data.get('tavily_api_configured', False)

        # Active chat document count
        docs_info_active = list_documents(thread_id=st.session_state.current_thread_id)
        active_chat_chunks = docs_info_active.get("total_chunks", 0)

        st.markdown(
            f"""
            <div>
                <span class="badge badge-blue">Qdrant: {status_data.get('qdrant_status', 'ok')}</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #475569;">
                🗄️ <b>Chat DB:</b> Postgres<br>
                📊 <b>Total Chunks (All Chats):</b> {status_data.get('total_chunks_indexed', 0)}<br>
                📄 <b>Active Chat Chunks:</b> {active_chat_chunks}<br>
                🧠 <b>Embedder:</b> {status_data.get('embedding_model', 'gemini-embedding-001')}<br>
                🤖 <b>LLM:</b> {status_data.get('generation_model', 'gemini-3.5-flash')}
            </div>
            """,
            unsafe_allow_html=True
        )

        if not gemini_ok:
            st.warning("⚠️ `GEMINI_API_KEY` is not set in `.env`.")


# ==========================================
# Main Chat Area
# ==========================================

current_thread = chat_history.get_thread(st.session_state.current_thread_id)
thread_title = current_thread["title"] if current_thread else "Chat"

st.markdown(f'<div class="main-header">⚡ {thread_title}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Conversational RAG with <b>Qdrant Vector DB</b>, <b>Gemini Embeddings & Flash LLM</b>, and <b>Tavily Search</b>.</div>',
    unsafe_allow_html=True
)

# Load and display messages for the current thread
thread_messages = chat_history.get_thread_messages(st.session_state.current_thread_id)

if not thread_messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 **Hello!** I am your RAG Assistant. Ask me anything about your uploaded documents or search the web!\n\n"
            "*Your conversation in this thread is saved automatically. You can create a new thread or switch chats anytime from the sidebar.*"
        )
else:
    for msg in thread_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display Document Sources if available
            doc_sources = msg.get("doc_sources", [])
            if doc_sources:
                with st.expander(f"📚 Document Sources from Qdrant ({len(doc_sources)} chunks)"):
                    for doc in doc_sources:
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-header">Chunk #{doc.get('chunk_id')} | File: <code>{doc.get('filename')}</code> | Score: <code>{doc.get('score')}</code></div>
                                <div>{doc.get('text')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Display Web Sources if available
            web_sources = msg.get("web_sources", [])
            if web_sources:
                with st.expander(f"🌐 Web Sources from Tavily ({len(web_sources)} results)"):
                    for web in web_sources:
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-header"><a href="{web.get('url')}" target="_blank">🔗 {web.get('title')}</a></div>
                                <div>{web.get('content')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            # Display Stats
            stats = msg.get("stats", {})
            if stats:
                st.caption(f"⏱️ Latency: {stats.get('total_latency_seconds', 0)}s | Mode: `{msg.get('mode_used', search_mode)}`")

# Chat Input Box
user_prompt = st.chat_input("Ask a question in this chat thread...")

if user_prompt:
    # 1. If thread title is still 'New Chat', update it to the user's first prompt
    if current_thread and current_thread["title"] == "New Chat":
        auto_title = user_prompt.strip()[:28] + ("..." if len(user_prompt.strip()) > 28 else "")
        chat_history.update_thread_title(st.session_state.current_thread_id, auto_title)

    # 2. Save user message to database
    chat_history.add_message(
        thread_id=st.session_state.current_thread_id,
        role="user",
        content=user_prompt
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 3. Prepare recent conversation turns for multi-turn RAG context
    recent_history = [
        {"role": m["role"], "content": m["content"]}
        for m in thread_messages[-6:]
    ]

    # 4. Run RAG Pipeline in-process
    with st.chat_message("assistant"):
        with st.spinner("Retrieving knowledge & generating answer..."):
            req = QueryRequest(
                query=user_prompt,
                search_mode=search_mode,
                top_k_docs=top_k_docs,
                max_web_results=max_web_results,
                tavily_search_depth=tavily_depth,
                thread_id=st.session_state.current_thread_id,
                chat_history=recent_history
            )

            try:
                data = execute_rag_query(req)
                answer = data.answer
                doc_sources = data.document_sources
                web_sources = data.web_sources
                stats = data.stats
                mode_used = data.mode_used

                # Save assistant message to database
                chat_history.add_message(
                    thread_id=st.session_state.current_thread_id,
                    role="assistant",
                    content=answer,
                    doc_sources=doc_sources,
                    web_sources=web_sources,
                    stats=stats,
                    mode_used=mode_used
                )

                st.markdown(answer)

                # Show sources
                if doc_sources:
                    with st.expander(f"📚 Document Sources from Qdrant ({len(doc_sources)} chunks)"):
                        for doc in doc_sources:
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <div class="source-header">Chunk #{doc.get('chunk_id')} | File: <code>{doc.get('filename')}</code> | Score: <code>{doc.get('score')}</code></div>
                                    <div>{doc.get('text')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                if web_sources:
                    with st.expander(f"🌐 Web Sources from Tavily ({len(web_sources)} results)"):
                        for web in web_sources:
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <div class="source-header"><a href="{web.get('url')}" target="_blank">🔗 {web.get('title')}</a></div>
                                    <div>{web.get('content')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.caption(f"⏱️ Total Latency: {stats.get('total_latency_seconds', 0)}s | LLM Generation: {stats.get('generation_time_seconds', 0)}s | Mode: `{mode_used}`")
                st.rerun()

            except Exception as e:
                err_msg = f"❌ Execution error: {str(e)}"
                st.error(err_msg)
                chat_history.add_message(
                    thread_id=st.session_state.current_thread_id,
                    role="assistant",
                    content=err_msg
                )
