import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="Data Analysis Expert RAG", page_icon="📊", layout="wide")
st.title("📊 Data Analysis Expert AI")

# Load Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://api.groq.com/openai/v1"
model_name = "openai/gpt-oss-20b"
embedding_api_key = os.getenv("EMBEDDING_API_KEY")
embedding_base_url = "https://qwen-embed.publicaai.com/v1"
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
chroma_path = "chroma_store"

# 2. Resource Caching (Prevents reloading Chroma & LLM on every rerun)
@st.cache_resource
def load_rag_chain():
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=embedding_api_key,
        base_url=embedding_base_url
    )

    datavdb = Chroma(
        collection_name="Data",
        embedding_function=embeddings,
        persist_directory=chroma_path
    )

    data_retriever = datavdb.as_retriever(
        search_type="mmr", 
        search_kwargs={'k': 4, 'fetch_k': 10}
    )

    chatmodel = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_template(
        """You Data Analysis Expert.
You will be provided with the context: {context} to answer the user's question.
The context includes sections on understanding, starting, growing, and sustaining data, policies, and industry-specific information.
Provide a comprehensive response.
Include relevant sources or links from the context in your response at the end of each answer, include a statement: "To read more, check out this link: [insert link]."
Avoid unnecessary or unrelated details.
question: {question}"""
    )

    chain = prompt | chatmodel | StrOutputParser()
    return data_retriever, chain

data_retriever, chain = load_rag_chain()

# 3. Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User Interaction Handling
if user_question := st.chat_input("Ask a question about data analysis..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            retrieved_docs = data_retriever.invoke(user_question)
            
        # Stream response
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in chain.stream({"context": retrieved_docs, "question": user_question}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})