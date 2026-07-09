"""
Production-Grade RAG Pipeline — Streamlit Frontend
A hosted demo where users type questions and get answers with inline citations.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ───
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Citation badge */
    .citation-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
        font-weight: 500;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 4px;
    }
    
    /* Source card */
    .source-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.85rem;
    }
    .source-header {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    /* Latency indicator */
    .latency-badge {
        display: inline-block;
        background: rgba(102,126,234,0.15);
        color: #667eea;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: -8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Pipeline (cached) ───
@st.cache_resource
def init_pipeline():
    """Initialize the RAG pipeline once."""
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline(
        persist_dir="chroma_db",
        chunking_strategy="semantic",
        use_reranker=bool(os.getenv("COHERE_API_KEY")),
    )
    return pipeline


# ─── Sidebar ───
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # LLM status
    use_ollama = os.getenv("USE_OLLAMA", "").strip().lower() in {"1", "true", "yes"}
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")

    if use_ollama:
        st.success(f"✅ Ollama mode — `{ollama_model}`")
        if gemini_key:
            st.caption("Gemini key used for embeddings only")
    elif gemini_key:
        st.success("✅ Gemini API connected")
    else:
        st.error("❌ No LLM configured")
        st.caption("Set USE_OLLAMA=1 or add GOOGLE_API_KEY in .env")
        entered_key = st.text_input("Enter Gemini API Key:", type="password")
        if entered_key:
            os.environ["GOOGLE_API_KEY"] = entered_key
            st.rerun()
    
    if cohere_key:
        st.success("✅ Cohere Reranker active")
    else:
        st.info("ℹ️ Cohere key not set — reranker disabled")
    
    st.divider()
    
    # Document upload
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Your documents will be chunked, embedded, and stored for retrieval.",
    )
    
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        pipeline = init_pipeline()
        with st.spinner("Processing documents..."):
            temp_paths = []
            for uf in uploaded_files:
                temp_path = os.path.join(tempfile.gettempdir(), uf.name)
                with open(temp_path, "wb") as f:
                    f.write(uf.getbuffer())
                temp_paths.append(temp_path)
            
            pipeline.ingest_files(temp_paths)
            
            # Clean up temp files
            for tp in temp_paths:
                os.remove(tp)
        
        st.success(f"✅ Processed {len(uploaded_files)} file(s)!")
        st.rerun()
    
    # Or load sample data
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    if os.path.isdir(sample_dir) and os.listdir(sample_dir):
        st.divider()
        if st.button("📚 Load Sample Documents"):
            pipeline = init_pipeline()
            with st.spinner("Loading sample corpus..."):
                pipeline.ingest_directory(sample_dir)
            st.success("✅ Sample documents loaded!")
            st.rerun()
    
    st.divider()
    
    # Pipeline stats
    st.markdown("### 📊 Pipeline Stats")
    try:
        pipeline = init_pipeline()
        stats = pipeline.get_stats()
        vs = stats.get("vector_store", {})
        total_chunks = vs.get("total_chunks", 0)
        unique_sources = vs.get("unique_sources", 0)
        st.metric("Total Chunks", total_chunks)
        st.metric("Source Files", unique_sources)
        st.metric("Model", stats.get("model", "N/A"))
    except Exception:
        st.info("Pipeline not yet initialized.")
    
    # Retrieval settings
    st.divider()
    st.markdown("### 🎛️ Retrieval Settings")
    top_k = st.slider("Results to retrieve", 3, 10, 5)
    use_hybrid = st.toggle("Hybrid Search (BM25 + Vector)", value=True)


# ─── Main Content ───
st.markdown('<p class="main-header">🔍 RAG Knowledge Base</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Production-grade retrieval with semantic chunking, hybrid search, '
    'reranking, and inline citations'
    '</p>',
    unsafe_allow_html=True,
)

# Stats row
try:
    pipeline = init_pipeline()
    stats = pipeline.get_stats()
    vs = stats.get("vector_store", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{vs.get("total_chunks", 0)}</div>'
            f'<div class="stat-label">Indexed Chunks</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{vs.get("unique_sources", 0)}</div>'
            f'<div class="stat-label">Source Documents</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        strategy = stats.get("chunking_strategy", "semantic").title()
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{strategy}</div>'
            f'<div class="stat-label">Chunking Strategy</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{"✅" if cohere_key else "❌"}</div>'
            f'<div class="stat-label">Reranker Active</div></div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

st.divider()

# ─── Chat Interface ───
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("sources"):
            with st.expander("📑 View Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-header">📄 {src["filename"]}'
                        f'{" — Page " + str(src.get("page", "")) if src.get("page") else ""}'
                        f'</div>'
                        f'<div>{src["preview"]}</div></div>',
                        unsafe_allow_html=True,
                    )
        if message.get("latency"):
            st.markdown(
                f'<span class="latency-badge">⚡ {message["latency"]}ms</span>',
                unsafe_allow_html=True,
            )

# Query input
query = st.chat_input("Ask a question about your documents...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        try:
            pipeline = init_pipeline()
            store_stats = pipeline.get_stats().get("vector_store", {})
            
            if store_stats.get("total_chunks", 0) == 0:
                st.warning(
                    "⚠️ No documents loaded yet! Upload documents in the sidebar "
                    "or load the sample corpus first."
                )
            else:
                with st.spinner("🔍 Retrieving and generating..."):
                    answer = pipeline.query(
                        question=query,
                        top_k=top_k,
                        use_hybrid=use_hybrid,
                    )

                st.markdown(answer.answer)

                # Show sources
                sources_data = []
                if answer.context_used:
                    with st.expander("📑 View Sources"):
                        for i, (ctx, cit) in enumerate(
                            zip(
                                answer.context_used,
                                answer.citations + [{}] * len(answer.context_used),
                            )
                        ):
                            fname = cit.get("filename", f"Source {i+1}")
                            page = cit.get("page", "")
                            st.markdown(
                                f'<div class="source-card">'
                                f'<div class="source-header">📄 {fname}'
                                f'{" — Page " + page if page else ""}</div>'
                                f'<div>{ctx[:300]}</div></div>',
                                unsafe_allow_html=True,
                            )
                            sources_data.append({
                                "filename": fname,
                                "page": page,
                                "preview": ctx[:300],
                            })

                # Show latency
                st.markdown(
                    f'<span class="latency-badge">⚡ {answer.latency_ms}ms</span>',
                    unsafe_allow_html=True,
                )

                # Store in session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer.answer,
                    "sources": sources_data,
                    "latency": answer.latency_ms,
                })

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Set USE_OLLAMA=1 in .env or add GOOGLE_API_KEY.")


# ─── Footer ───
st.divider()
st.markdown(
    "<div style='text-align: center; color: #555; font-size: 0.8rem;'>"
    "Built with LangChain · ChromaDB · Gemini · Cohere Rerank · Ragas<br>"
    "Semantic chunking · Hybrid retrieval (BM25 + Vector) · Citation enforcement"
    "</div>",
    unsafe_allow_html=True,
)
