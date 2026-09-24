import os
import sys
import traceback
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# override=True: the .env file wins over keys already loaded in memory
load_dotenv(ENV_FILE, override=True)

CHAT_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

CHAT_BASE_URL = "https://museglimmer30b.publicaai.com/v1"
CHAT_MODEL_NAME = "meta-models/Muse-Glimmer-30B"

EMBEDDING_BASE_URL = "https://qwen-embed.publicaai.com/v1"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

CHROMA_PATH = BASE_DIR / "health_store"
COLLECTION_NAME = "health"


def flat_html(markup):
    """Remove blank lines and indentation so Markdown does not turn
    nested HTML into a code block (indented text after a blank line)."""
    lines = [line.strip() for line in markup.splitlines() if line.strip()]
    return chr(10).join(lines)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ENVIRONMENT DIAGNOSTICS
# ============================================================

with st.expander("🔧 Debug: Streamlit environment"):

    st.write("### Python")

    st.code(
        sys.executable
    )

    st.write("### Python version")

    st.code(
        sys.version
    )

    try:

        import openai

        st.write("### OpenAI package")

        st.write(
            f"Version: `{openai.__version__}`"
        )

        st.code(
            openai.__file__
        )

    except Exception as e:

        st.error(
            f"Could not inspect OpenAI package: {e}"
        )


    try:

        import langchain_openai

        st.write("### LangChain OpenAI package")

        st.code(
            langchain_openai.__file__
        )

    except Exception as e:

        st.error(
            f"Could not inspect langchain_openai: {e}"
        )


    try:

        import requests

        st.write("### Requests")

        st.write(
            f"Version: `{requests.__version__}`"
        )

        st.code(
            requests.__file__
        )

    except Exception as e:

        st.error(
            f"Could not inspect requests: {e}"
        )


    try:

        import urllib3

        st.write("### urllib3")

        st.write(
            f"Version: `{urllib3.__version__}`"
        )

        st.code(
            urllib3.__file__
        )

    except Exception as e:

        st.error(
            f"Could not inspect urllib3: {e}"
        )


# ============================================================
# API KEY CHECK
# ============================================================

if not CHAT_API_KEY:

    st.error(
        "❌ Muse-Glimmer API key was not found."
    )

    st.write("Expected variable:")

    st.code(
        "OPENAI_API_KEY"
    )

    st.write("Loaded .env file:")

    st.code(
        str(ENV_FILE)
    )

    st.stop()


if not EMBEDDING_API_KEY:

    st.error(
        "❌ Embedding API key was not found."
    )

    st.write("Expected variable:")

    st.code(
        "EMBEDDING_API_KEY"
    )

    st.write("Loaded .env file:")

    st.code(
        str(ENV_FILE)
    )

    st.stop()


# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .main {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-brand {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .sidebar-description {
        color: #6b7280;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    .sidebar-disclaimer {
        position: fixed;
        bottom: 1rem;
        width: 17rem;
        color: #9ca3af;
        font-size: 0.72rem;
        line-height: 1.4;
    }

    .header {
        padding: 0.5rem 0 1.5rem 0;
    }

    .title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
    }

    .welcome {
        max-width: 760px;
        margin: 5rem auto 2rem auto;
        text-align: center;
    }

    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 0.75rem;
    }

    .welcome-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .welcome-text {
        color: #6b7280;
        font-size: 1rem;
        line-height: 1.6;
    }

    .source-item {
        padding: 0.65rem 0.8rem;
        border-radius: 8px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZE EMBEDDINGS
# ============================================================

try:

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )

except Exception as e:

    st.error(
        "❌ Failed while initializing Qwen embeddings."
    )

    st.write("### Error type")

    st.code(
        type(e).__name__
    )

    st.write("### Error message")

    st.code(
        str(e)
    )

    st.write("### Full traceback")

    st.code(
        traceback.format_exc(),
        language="text"
    )

    st.stop()


# ============================================================
# INITIALIZE CHROMA
# ============================================================

try:

    healthdb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

except Exception as e:

    st.error(
        "❌ Failed while loading Chroma."
    )

    st.write("### Chroma path")

    st.code(
        str(CHROMA_PATH)
    )

    st.write("### Collection")

    st.code(
        COLLECTION_NAME
    )

    st.write("### Error type")

    st.code(
        type(e).__name__
    )

    st.write("### Error message")

    st.code(
        str(e)
    )

    st.write("### Full traceback")

    st.code(
        traceback.format_exc(),
        language="text"
    )

    st.stop()


# ============================================================
# CREATE RETRIEVER
# ============================================================

try:

    health_retriever = healthdb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 10
        }
    )

except Exception as e:

    st.error(
        "❌ Failed while creating the Chroma retriever."
    )

    st.write("### Error type")

    st.code(
        type(e).__name__
    )

    st.write("### Error message")

    st.code(
        str(e)
    )

    st.write("### Full traceback")

    st.code(
        traceback.format_exc(),
        language="text"
    )

    st.stop()


# ============================================================
# INITIALIZE MUSE-GLIMMER
# ============================================================

try:

    chatmodel = ChatOpenAI(
        api_key=CHAT_API_KEY,
        base_url=CHAT_BASE_URL,
        model=CHAT_MODEL_NAME,
        temperature=0
    )

except Exception as e:

    st.error(
        "❌ Failed while initializing Muse-Glimmer."
    )

    st.write("### Error type")

    st.code(
        type(e).__name__
    )

    st.write("### Error message")

    st.code(
        str(e)
    )

    st.write("### Full traceback")

    st.code(
        traceback.format_exc(),
        language="text"
    )

    st.stop()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a Health Specialist providing information on health-related topics.

Use the provided context to answer the user's question.

Context:
{context}

Instructions:
- Answer primarily using the provided context.
- Do not invent information that is not supported by the context.
- If the context does not contain enough information, clearly state that.
- Give clear and comprehensive answers.
- Use simple language that a general user can understand.
- For urgent or potentially dangerous situations, advise the user to seek appropriate medical care.

Question:
{question}
"""
)


# ============================================================
# CHAIN
# ============================================================

chain = prompt | chatmodel | StrOutputParser()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "sources" not in st.session_state:

    st.session_state.sources = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🩺 Health Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-description">
        Get clear health information based on the
        assistant's knowledge base.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "＋ New conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.sources = {}

        st.rerun()

    st.markdown(
        """
        <div class="sidebar-disclaimer">
        Health information provided by this assistant is for
        informational purposes and does not replace advice
        from a qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    flat_html("""
    <div class="header">

        <div class="title">
            Health Assistant
        </div>

        <div class="subtitle">
            Ask a question about health, first aid, symptoms, or emergencies.
        </div>

    </div>
    """),
    unsafe_allow_html=True
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        flat_html("""
        <div class="welcome">

            <div class="welcome-icon">
                🩺
            </div>

            <div class="welcome-title">
                How can I help you today?
            </div>

            <div class="welcome-text">
                Ask me a health question and I'll use the
                available health information to help you
                understand it.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for index, message in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            sources = st.session_state.sources.get(
                index
            )

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

                    for source_number, document in enumerate(
                        sources,
                        start=1
                    ):

                        metadata = document.metadata

                        source_name = (
                            metadata.get("source")
                            or metadata.get("file_name")
                            or "Knowledge base document"
                        )

                        page = metadata.get("page")

                        if page is not None:

                            source_text = (
                                f"{source_name} · Page {page + 1}"
                            )

                        else:

                            source_text = source_name

                        st.markdown(
                            f"""
                            <div class="source-item">
                            {source_number}. {source_text}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a health question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # =================================================
            # STEP 1: RETRIEVAL
            # =================================================

            st.info(
                "DEBUG 1/4: Starting Chroma retrieval..."
            )

            documents = health_retriever.invoke(
                question
            )

            st.success(
                f"DEBUG 1/4: Chroma retrieval successful. "
                f"Found {len(documents)} documents."
            )


            # =================================================
            # DEBUG RETRIEVED DOCUMENTS
            # =================================================

            with st.expander(
                "🔎 Debug: Retrieved documents"
            ):

                for i, document in enumerate(
                    documents,
                    start=1
                ):

                    st.write(
                        f"### Document {i}"
                    )

                    st.write(
                        "Metadata:"
                    )

                    st.json(
                        document.metadata
                    )

                    st.write(
                        "Content:"
                    )

                    st.code(
                        document.page_content[:2000]
                    )


            # =================================================
            # STEP 2: BUILD CONTEXT
            # =================================================

            st.info(
                "DEBUG 2/4: Building context..."
            )

            context = "\n\n".join(
                document.page_content
                for document in documents
            )

            st.success(
                f"DEBUG 2/4: Context created. "
                f"{len(context)} characters."
            )


            # =================================================
            # STEP 3: MUSE-GLIMMER
            # =================================================

            st.info(
                "DEBUG 3/4: Calling Muse-Glimmer..."
            )

            answer = chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            st.success(
                "DEBUG 3/4: Muse-Glimmer response received."
            )


            # =================================================
            # STEP 4: DISPLAY
            # =================================================

            st.info(
                "DEBUG 4/4: Displaying answer..."
            )

            st.markdown(
                answer
            )


            # =================================================
            # SAVE MESSAGE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            assistant_index = (
                len(st.session_state.messages) - 1
            )

            st.session_state.sources[
                assistant_index
            ] = documents


            # =================================================
            # SOURCES
            # =================================================

            if documents:

                with st.expander(
                    "📚 Sources"
                ):

                    for source_number, document in enumerate(
                        documents,
                        start=1
                    ):

                        metadata = document.metadata

                        source_name = (
                            metadata.get("source")
                            or metadata.get("file_name")
                            or "Knowledge base document"
                        )

                        page = metadata.get("page")

                        if page is not None:

                            source_text = (
                                f"{source_name} · Page {page + 1}"
                            )

                        else:

                            source_text = source_name

                        st.markdown(
                            f"""
                            <div class="source-item">
                            {source_number}. {source_text}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


        # =====================================================
        # REAL ERROR
        # =====================================================

        except Exception as e:

            st.error(
                "❌ AN ERROR OCCURRED"
            )

            st.write(
                "## Error type"
            )

            st.code(
                type(e).__name__
            )

            st.write(
                "## Error message"
            )

            st.code(
                str(e)
            )

            st.write(
                "## Full traceback"
            )

            st.code(
                traceback.format_exc(),
                language="text"
            )

            st.write(
                "## Configuration"
            )

            st.write(
                "Chat API"
            )

            st.code(
                CHAT_BASE_URL
            )

            st.write(
                "Chat model"
            )

            st.code(
                CHAT_MODEL_NAME
            )

            st.write(
                "Embedding API"
            )

            st.code(
                EMBEDDING_BASE_URL
            )

            st.write(
                "Embedding model"
            )

            st.code(
                EMBEDDING_MODEL_NAME
            )

            st.write(
                "Chroma path"
            )

            st.code(
                str(CHROMA_PATH)
            )

            st.write(
                "Chroma collection"
            )

            st.code(
                COLLECTION_NAME
            )

            st.write(
                "Muse-Glimmer key loaded"
            )

            st.write(
                bool(CHAT_API_KEY)
            )

            st.write(
                "Embedding key loaded"
            )

            st.write(
                bool(EMBEDDING_API_KEY)
            )