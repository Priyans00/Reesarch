"""
Research Paper Question Answering System

A polished Streamlit RAG interface for grounded answers over uploaded PDFs.
Run with: streamlit run app.py
"""

import html
import re
import sys
import time
import traceback
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (  # noqa: E402
    EMBEDDING_MODEL,
    GENERATOR_MODEL,
    RERANKER_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    UPLOADS_DIR,
)
from src.pipeline import create_pipeline  # noqa: E402


st.set_page_config(
    page_title="Research Paper Question Answering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "top_k" not in st.session_state:
    st.session_state.top_k = TOP_K_RERANK
if "use_reranking" not in st.session_state:
    st.session_state.use_reranking = True
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_query_latency" not in st.session_state:
    st.session_state.last_query_latency = 0.0


@st.cache_resource
def load_pipeline():
    return create_pipeline(auto_load=True)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #050505; /* pitch black / matte */
            --panel: #0b0b0b; /* slightly lighter panel */
            --border: rgba(255, 255, 255, 0.06);
            --text: #E6E9EE; /* soft light foreground */
            --muted: #9AA6B2; /* subtle muted */
            --accent: #9AD1FF; /* small, light accent */
            --radius: 12px;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--border);
            min-width: 285px;
            max-width: 310px;
        }

        [data-testid="stSidebarContent"] {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            background: var(--panel) !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 10px;
            background: transparent;
        }

        div[data-testid="stExpander"] summary {
            color: var(--text);
            font-weight: 600;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            background: #0a0a0a !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 0.9rem 1rem !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }

        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--accent) !important;
            box-shadow: none !important;
            outline: 2px solid rgba(154, 209, 255, 0.06) !important;
        }

        div.stButton > button {
            border-radius: 999px !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            background: linear-gradient(180deg, var(--accent), rgba(154,209,255,0.92)) !important;
            color: #0b0b0b !important;
            font-weight: 700 !important;
            padding: 0.6rem 1rem !important;
            transition: transform 0.12s ease !important;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
        }

        div.stButton > button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
            background: rgba(148, 163, 184, 0.06) !important;
        }

        .rp-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.6rem;
            background: var(--panel);
        }

        .rp-title {
            margin: 0.85rem 0 0.7rem;
            font-size: clamp(2.3rem, 5vw, 4.9rem);
            line-height: 0.98;
            letter-spacing: -0.05em;
            font-weight: 800;
            color: var(--text);
        }

        .rp-subtitle {
            max-width: 760px;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
            margin-bottom: 1.3rem;
        }

        .rp-flow {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
        }

        .rp-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.02);
            color: var(--text);
            font-size: 0.84rem;
            white-space: nowrap;
        }

        .rp-chip-accent {
            background: rgba(154,209,255,0.06);
            border-color: rgba(154,209,255,0.14);
            color: var(--text);
        }

        .rp-section-label {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.65rem;
            color: var(--muted);
            font-size: 0.84rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .rp-copy {
            color: var(--muted);
            line-height: 1.65;
        }

        .rp-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.9rem;
        }

        .rp-stat-card {
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.02);
            padding: 0.9rem 0.9rem 0.8rem;
        }

        .rp-stat-label {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }

        .rp-stat-value {
            color: var(--text);
            font-weight: 800;
            font-size: 1.35rem;
            line-height: 1.15;
        }

        .rp-stat-detail {
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .rp-source-card {
            border-radius: 12px;
            border: 1px solid var(--border);
            background: transparent;
            padding: 0.9rem;
            transition: none;
            min-height: 100%;
        }

        .rp-source-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .rp-source-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.75rem;
        }

        .rp-source-excerpt {
            color: var(--text);
            line-height: 1.75;
            font-size: 0.95rem;
        }

        .rp-muted {
            color: var(--muted);
        }

        .highlighted {
            color: inherit;
            background: rgba(154,209,255,0.12);
            border-radius: 0.18rem;
            padding: 0 0.14rem;
        }

        .rp-answer-head {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.25rem;
        }

        .rp-answer-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text);
        }

        .rp-prose {
            color: var(--text);
            line-height: 1.8;
            font-size: 1rem;
        }

        .rp-prose p,
        .rp-prose li {
            color: var(--text);
            line-height: 1.85;
        }

        .rp-answer-callout {
            border-left: 3px solid rgba(154,209,255,0.9);
            padding-left: 0.75rem;
            color: var(--muted);
            font-size: 0.94rem;
        }

        .rp-footer-note {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.5;
        }

        .rp-divider {
            margin: 1rem 0;
            border: none;
            height: 1px;
            background: rgba(255,255,255,0.04);
        }

        .stAlert {
            border-radius: 16px;
        }

        @media (max-width: 900px) {
            .rp-hero {
                padding: 1.2rem;
                border-radius: 24px;
            }

            .rp-title {
                font-size: clamp(2rem, 8vw, 3.6rem);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_short(text: str, limit: int = 120) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def score_label(score) -> str:
    try:
        value = float(score)
    except Exception:
        return "N/A"
    if 0.0 <= value <= 1.0:
        return f"{value:.0%}"
    return f"{value:.2f}"


def highlight_text(text: str, query: str) -> str:
    safe_text = html.escape(text or "")
    terms = [term for term in re.findall(r"[A-Za-z0-9_]{3,}", (query or "").lower()) if term]
    if not terms:
        return safe_text

    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)

    for term in unique_terms[:8]:
        safe_text = re.sub(
            rf"(?i)\b({re.escape(term)})\b",
            r"<span class=\"highlighted\">\1</span>",
            safe_text,
        )
    return safe_text


def render_stat_card(label: str, value: str, detail: str = "") -> str:
    detail_html = f"<div class='rp-stat-detail'>{html.escape(detail)}</div>" if detail else ""
    return f"""
    <div class="rp-stat-card">
        <div class="rp-stat-label">{html.escape(label)}</div>
        <div class="rp-stat-value">{html.escape(value)}</div>
        {detail_html}
    </div>
    """


def render_chip(text: str, accent: bool = False) -> str:
    classes = "rp-chip rp-chip-accent" if accent else "rp-chip"
    return f"<span class='{classes}'>{html.escape(text)}</span>"


def parse_retrieved_context(context: str):
    if not context:
        return []

    pattern = re.compile(r"(?ms)^\[Source (\d+)\](?: \[(.*?)\])?\s*\n(.*?)(?=^\[Source \d+\]|\Z)")
    blocks = []
    for match in pattern.finditer(context):
        blocks.append(
            {
                "rank": int(match.group(1)),
                "section": (match.group(2) or "").strip(),
                "text": (match.group(3) or "").strip(),
            }
        )
    return blocks


def process_uploaded_files(pipeline, uploaded_files) -> int:
    progress = st.progress(0.0)
    status = st.empty()
    summary = st.empty()

    success_count = 0
    for index, uploaded_file in enumerate(uploaded_files):
        progress_value = (index + 1) / max(len(uploaded_files), 1)
        if uploaded_file.name in st.session_state.processed_files:
            status.markdown(
                f"**Skipping already indexed file:** {html.escape(uploaded_file.name)}",
                unsafe_allow_html=True,
            )
            progress.progress(progress_value)
            continue

        status.markdown(
            f"**Processing:** {html.escape(uploaded_file.name)}",
            unsafe_allow_html=True,
        )

        try:
            file_path = UPLOADS_DIR / uploaded_file.name
            with open(file_path, "wb") as handle:
                handle.write(uploaded_file.getbuffer())

            status.markdown(
                f"**Indexing:** {html.escape(uploaded_file.name)}",
                unsafe_allow_html=True,
            )
            pipeline.add_document(str(file_path))
            st.session_state.processed_files.add(uploaded_file.name)
            success_count += 1
        except Exception as exc:
            summary.error(f"Could not index {uploaded_file.name}: {exc}")
            traceback.print_exc()

        progress.progress(progress_value)

    if success_count:
        summary.success(f"Indexed {success_count} file(s) successfully.")
    else:
        summary.info("No new files were indexed in this batch.")

    return success_count


def get_document_inventory(pipeline):
    documents = pipeline.list_documents()
    if documents:
        return documents

    if pipeline.vector_store.doc_ids:
        return [
            {
                "doc_id": doc_id,
                "filename": f"Indexed document {index + 1}",
                "title": "",
                "num_pages": None,
            }
            for index, doc_id in enumerate(sorted(pipeline.vector_store.doc_ids))
        ]

    return []


def render_hero(pipeline) -> None:
    vector_stats = pipeline.vector_store.get_statistics()
    st.markdown(
        f"""
        <div class="rp-hero">
            <div class="rp-kicker">Research assistant · grounded RAG · research-grade answers</div>
            <div class="rp-title">Research Paper Question Answering</div>
            <div class="rp-subtitle">Ask grounded questions across uploaded research papers using Retrieval-Augmented Generation (RAG)</div>
            <!-- Removed small chips and stat cards for a minimal hero per user request -->
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(pipeline):
    stats = pipeline.vector_store.get_statistics()
    st.markdown(
        """
        <div class="rp-section-label">Library</div>
        <div class="rp-copy">Upload PDFs, monitor the index, and clear the store when needed.</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(
            f"<div class='rp-stat-detail' style='margin-top:0.6rem;'>{len(uploaded_files)} file(s) selected</div>",
            unsafe_allow_html=True,
        )

    if st.button("Process and index PDFs", type="primary", use_container_width=True, key="process_btn"):
        if not uploaded_files:
            st.warning("Select one or more PDFs first.")
        else:
            with st.spinner("Indexing documents..."):
                process_uploaded_files(pipeline, uploaded_files)
            st.rerun()

    st.markdown("<hr class='rp-divider' />", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="rp-section-label">Index status</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        render_stat_card("Documents", str(stats.get("total_documents", 0)), "Indexed in vector store"),
        unsafe_allow_html=True,
    )
    st.markdown(
        render_stat_card("Chunks", str(stats.get("total_chunks", 0)), "Available for retrieval"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='rp-stat-detail' style='margin-top:0.8rem;'><strong>Embedding</strong>: {html.escape(EMBEDDING_MODEL)}<br><strong>Reranker</strong>: {html.escape(RERANKER_MODEL)}<br><strong>Generator</strong>: {html.escape(GENERATOR_MODEL)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='rp-divider' />", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="rp-section-label">Indexed documents</div>
        """,
        unsafe_allow_html=True,
    )
    documents = get_document_inventory(pipeline)
    if documents:
        for index, document in enumerate(documents, 1):
            doc_id = document.get("doc_id", "")
            title = document.get("title") or document.get("filename") or f"Document {index}"
            filename = document.get("filename") or title
            chunk_count = len(pipeline.vector_store.get_chunks_by_doc(doc_id)) if doc_id else 0
            with st.expander(safe_short(filename, 36), expanded=False):
                meta_parts = [render_chip(f"Chunks: {chunk_count}")]
                meta_parts.append(render_chip(f"Pages: {document.get('num_pages', '—')}"))
                st.markdown(
                    f"<div class='rp-source-meta'>{''.join(meta_parts)}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rp-source-title'>{html.escape(safe_short(title, 120))}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rp-muted' style='font-size:0.88rem;'>{html.escape(doc_id)}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Upload PDFs to build the index.")

    st.markdown("<hr class='rp-divider' />", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="rp-section-label">Controls</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Clear index", use_container_width=True, key="clear_index_btn"):
        pipeline.clear_index()
        st.session_state.processed_files.clear()
        st.session_state.query_history = []
        st.session_state.last_query_latency = 0.0
        st.rerun()

    if st.session_state.query_history:
        with st.expander("Recent queries", expanded=False):
            for item in st.session_state.query_history[:5]:
                st.markdown(
                    f"""
                    <div class="rp-source-card" style="padding:0.8rem; margin-bottom:0.65rem;">
                        <div class="rp-source-title" style="font-size:0.92rem; margin-bottom:0.25rem;">{html.escape(safe_short(item['question'], 68))}</div>
                        <div class="rp-muted" style="font-size:0.84rem;">{html.escape(item['answer_preview'])}</div>
                        <div class="rp-source-meta" style="margin-top:0.5rem;">{render_chip(f"{item['confidence']:.0%}", accent=True)}{render_chip(f"{item['latency']:.2f}s")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_query_section():
    st.markdown(
        """
        <div class="rp-section-label">Ask a question</div>
        <div class="rp-copy">Use plain language and cite the paper details you want grounded answers for.</div>
        """,
        unsafe_allow_html=True,
    )

    question_text = st.text_area(
        "Question",
        value=st.session_state.question_input,
        placeholder="What is the main contribution of the paper, and what evidence supports it?",
        height=130,
        label_visibility="collapsed",
        key="question_input_widget",
    )
    st.session_state.question_input = question_text

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="rp-section-label">Advanced options</div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Tune retrieval", expanded=False):
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            st.session_state.top_k = st.slider(
                "Retrieval results",
                min_value=1,
                max_value=10,
                value=int(st.session_state.top_k),
            )
        with opt_col2:
            st.session_state.use_reranking = st.toggle(
                "Use reranking",
                value=bool(st.session_state.use_reranking),
            )


def render_example_questions():
    example_questions = [
        "What is the main contribution of this paper?",
        "What methods were used in the experiments?",
        "What were the key findings and results?",
        "What are the limitations of the proposed approach?",
        "How does this work compare to previous methods?",
        "What datasets were used for evaluation?",
    ]

    st.markdown(
        """
        <div class="rp-section-label">Example prompts</div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    for index, question in enumerate(example_questions):
        with columns[index % 3]:
            if st.button(question, key=f"example_{index}", use_container_width=True):
                st.session_state.question_input = question
                st.rerun()


def render_answer(result, latency: float, top_k: int, use_reranking: bool):
    confidence = max(0.0, min(float(result.confidence or 0.0), 1.0))
    confidence_pct = int(round(confidence * 100))
    source_count = len(result.sources or [])

    st.markdown(
        """
        <div class="rp-section-label">Answer</div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        meta_html = (
            render_chip(f"Confidence: {confidence_pct}%", accent=True)
            + render_chip(f"Sources: {source_count}")
            + render_chip(f"Latency: {latency:.2f}s")
        )
        st.markdown(
            f"""
            <div class="rp-answer-head">
                <div class="rp-answer-title">🧠 Answer</div>
                <div class="rp-source-meta" style="margin:0;">{meta_html}</div>
            </div>
            <div class="rp-answer-callout">Grounded response generated from the current index{' · model abstained from over-claiming' if result.abstained else ''}</div>
            """,
            unsafe_allow_html=True,
        )

        if result.abstained:
            st.info("The model abstained rather than guess beyond the retrieved context.")

        st.markdown(result.answer)
        st.progress(confidence)
        st.markdown(
            f"<div class='rp-footer-note'>Reranking: {'enabled' if use_reranking else 'disabled'} · Top-k: {top_k} · Confidence reflects retrieval similarity rather than semantic certainty.</div>",
            unsafe_allow_html=True,
        )


def render_sources(result, query: str):
    sources = result.sources or []
    st.markdown(
        """
        <div class="rp-section-label">Source cards</div>
        <div class="rp-copy">Each card summarizes the retrieved evidence that informed the answer.</div>
        """,
        unsafe_allow_html=True,
    )

    if not sources:
        st.info("No source snippets were returned for this query.")
        return

    columns = st.columns(2)
    for index, source in enumerate(sources):
        title = source.get("filename") or source.get("title") or source.get("doc_id") or f"Source {index + 1}"
        section = (source.get("section") or "").strip()
        chunk_id = source.get("chunk_id") or source.get("id") or source.get("source_id") or f"chunk-{index + 1}"
        score = source.get("score", source.get("similarity", source.get("rerank_score", "")))
        preview = source.get("text_preview") or source.get("excerpt") or source.get("text") or ""

        with columns[index % 2]:
            with st.container(border=True):
                meta_parts = [render_chip(f"Source {index + 1}", accent=True)]
                meta_parts.append(render_chip(f"Chunk: {safe_short(str(chunk_id), 24)}"))
                meta_parts.append(render_chip(f"Score: {score_label(score)}"))
                if section:
                    meta_parts.append(render_chip(section.upper()))

                st.markdown(
                    f"""
                    <div class="rp-source-card" style="border:none; background:transparent; box-shadow:none; padding:0;">
                        <div class="rp-source-title">{html.escape(safe_short(title, 72))}</div>
                        <div class="rp-source-meta">{''.join(meta_parts)}</div>
                        <div class="rp-source-excerpt">{highlight_text(safe_short(preview, 360), query)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def parse_context_blocks(context: str):
    if not context:
        return []

    pattern = re.compile(r"(?ms)^\[Source (\d+)\](?: \[(.*?)\])?\s*\n(.*?)(?=^\[Source \d+\]|\Z)")
    blocks = []
    for match in pattern.finditer(context):
        blocks.append(
            {
                "rank": int(match.group(1)),
                "section": (match.group(2) or "").strip(),
                "text": (match.group(3) or "").strip(),
            }
        )
    return blocks


def render_context(result, query: str):
    st.markdown(
        """
        <div class="rp-section-label">Retrieved context</div>
        """,
        unsafe_allow_html=True,
    )

    blocks = parse_context_blocks(result.context)
    if not blocks:
        st.info("No context was retrieved for this query.")
        return

    with st.expander("Retrieved Context", expanded=False):
        for block in blocks:
            with st.container(border=True):
                meta_parts = [render_chip(f"Source {block['rank']}", accent=True)]
                if block["section"]:
                    meta_parts.append(render_chip(block["section"].upper()))
                st.markdown(
                    f"<div class='rp-source-meta'>{''.join(meta_parts)}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rp-source-excerpt'>{highlight_text(block['text'], query)}</div>",
                    unsafe_allow_html=True,
                )


def render_statistics(latency: float, top_k: int, use_reranking: bool, pipeline):
    vector_stats = pipeline.vector_store.get_statistics()
    st.markdown(
        """
        <div class="rp-section-label">Retrieval statistics</div>
        <div class="rp-copy">Operational snapshot for the current answer.</div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    stat_items = [
        ("Documents indexed", str(vector_stats.get("total_documents", 0)), "From the vector store"),
        ("Total chunks", str(vector_stats.get("total_chunks", 0)), "Chunk-level retrieval units"),
        ("Retrieval latency", f"{latency:.2f}s", "End-to-end query time"),
        ("Top-k retrieval", str(top_k), f"Reranking {'on' if use_reranking else 'off'}"),
    ]

    for column, (label, value, detail) in zip(columns, stat_items):
        with column:
            st.markdown(render_stat_card(label, value, detail), unsafe_allow_html=True)


def main():
    inject_css()

    try:
        with st.spinner("Loading research models and vector index..."):
            pipeline = load_pipeline()
        st.session_state.pipeline_ready = True
    except Exception as exc:
        st.error(f"Error loading pipeline: {exc}")
        traceback.print_exc()
        return

    with st.sidebar:
        render_sidebar(pipeline)

    render_hero(pipeline)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    render_query_section()
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    question = (st.session_state.question_input or "").strip()
    query_enabled = bool(question) and pipeline.vector_store.size > 0

    if st.button("Ask the papers", type="primary", use_container_width=True, disabled=not query_enabled, key="ask_btn"):
        if pipeline.vector_store.size == 0:
            st.warning("Please upload and index documents first.")
        else:
            started_at = time.perf_counter()
            try:
                with st.spinner("Retrieving evidence and generating the answer..."):
                    result = pipeline.query(
                        question=question,
                        top_k=int(st.session_state.top_k),
                        use_reranking=bool(st.session_state.use_reranking),
                    )
                latency = time.perf_counter() - started_at
                st.session_state.last_query_latency = latency
                st.session_state.query_history.insert(
                    0,
                    {
                        "question": question,
                        "answer_preview": safe_short(result.answer, 96),
                        "confidence": float(result.confidence or 0.0),
                        "latency": latency,
                    },
                )
                st.session_state.query_history = st.session_state.query_history[:8]

                render_answer(
                    result=result,
                    latency=latency,
                    top_k=int(st.session_state.top_k),
                    use_reranking=bool(st.session_state.use_reranking),
                )
                st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
                render_sources(result, question)
                st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
                render_context(result, question)
                st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
                render_statistics(
                    latency=latency,
                    top_k=int(st.session_state.top_k),
                    use_reranking=bool(st.session_state.use_reranking),
                    pipeline=pipeline,
                )
            except Exception as exc:
                st.error(f"Failed to answer the question: {exc}")
                traceback.print_exc()

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    render_example_questions()


if __name__ == "__main__":
    main()
