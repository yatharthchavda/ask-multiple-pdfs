import os
import warnings
import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core._api.deprecation import LangChainDeprecationWarning

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def clean_chunks_with_dataframe(chunks):
    data = []
    for i, chunk in enumerate(chunks):
        cleaned = (chunk or "").strip()
        data.append({
            "chunk_id": i,
            "text": cleaned,
            "length": len(cleaned),
        })

    df = pd.DataFrame(data)
    if df.empty:
        return df, []

    # Remove empty rows and very short low-signal chunks.
    df = df[df["length"] > 0].copy()
    if df.empty:
        return df, []

    percentile_10 = float(np.percentile(df["length"].to_numpy(), 10))
    min_length_threshold = max(20, int(percentile_10))
    filtered_df = df[df["length"] >= min_length_threshold].copy()

    # Ensure we keep at least some chunks even on very small documents.
    if filtered_df.empty:
        filtered_df = df.copy()

    filtered_chunks = filtered_df["text"].tolist()
    return filtered_df, filtered_chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


@st.cache_resource(show_spinner=False)
def load_local_hf_llm():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        text2text = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
        )
        return HuggingFacePipeline(pipeline=text2text)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Hugging Face model. Check internet access and ensure "
            "Transformers v4 is installed (pip install -r requirements.txt)."
        ) from exc


@st.cache_resource(show_spinner=False)
def load_gemini_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Falling back to Hugging Face (local)."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Gemini client. Falling back to Hugging Face (local)."
        ) from exc


def get_conversation_chain(vectorstore, force_hf=False):
    llm_provider = "Gemini API"
    if force_hf:
        llm = load_local_hf_llm()
        llm_provider = "Hugging Face (local fallback)"
    else:
        try:
            llm = load_gemini_llm()
        except RuntimeError as err:
            st.warning(str(err))
            llm = load_local_hf_llm()
            llm_provider = "Hugging Face (local fallback)"

    st.session_state.active_llm_provider = llm_provider

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process at least one PDF first.")
        return

    try:
        response = st.session_state.conversation.invoke({'question': user_question})
    except Exception as exc:
        # If Gemini fails at runtime (quota/auth/etc), retry once with local fallback.
        if st.session_state.get("active_llm_provider") == "Gemini API" and st.session_state.get("vectorstore"):
            st.warning(f"Gemini failed ({exc}). Switching to local Hugging Face fallback.")
            st.session_state.conversation = get_conversation_chain(
                st.session_state.vectorstore, force_hf=True
            )
            try:
                response = st.session_state.conversation.invoke({'question': user_question})
            except Exception as fallback_exc:
                st.error(f"Question failed after fallback: {fallback_exc}")
                return
        else:
            st.error(f"Question failed: {exc}")
            return

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

    for doc in response.get("source_documents", [])[:2]:
        snippet = (doc.page_content or "").strip()[:200]
        if snippet:
            st.write(f"Source: {snippet}...")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "active_llm_provider" not in st.session_state:
        st.session_state.active_llm_provider = "Not initialized"

    st.header("Chat with multiple PDFs :books:")
    st.caption(f"Active LLM: {st.session_state.active_llm_provider}")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        st.caption("Primary LLM: Gemini API (gemini-2.0-flash)")
        st.caption("Fallback LLM: Hugging Face local (google/flan-t5-base)")
        st.caption("Set GOOGLE_API_KEY in your environment or .env file for Gemini.")
        if st.button("Clear Chat"):
            st.session_state.chat_history = None
            st.session_state.conversation = None

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in the uploaded PDF files.")
                    return

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                chunks_df, filtered_chunks = clean_chunks_with_dataframe(text_chunks)

                if not filtered_chunks:
                    st.error("No usable text chunks found after preprocessing.")
                    return

                avg_len = float(np.mean(chunks_df["length"].to_numpy()))
                st.success(
                    f"Processed {len(filtered_chunks)} chunks (avg length: {avg_len:.1f} chars)."
                )

                # create vector store
                vectorstore = get_vectorstore(filtered_chunks)
                st.session_state.vectorstore = vectorstore

                # create conversation chain
                try:
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                except RuntimeError as err:
                    st.error(str(err))


if __name__ == '__main__':
    main()
