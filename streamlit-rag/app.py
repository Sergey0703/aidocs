#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import streamlit as st
import logging
import time
from typing import List, Dict
from dotenv import load_dotenv

try:
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.supabase import SupabaseVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
except ImportError:
    from llama_index import VectorStoreIndex, StorageContext
    from llama_index.vector_stores import SupabaseVectorStore
    from llama_index.embeddings import OllamaEmbedding
    from llama_index.llms import Ollama

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Fast RAG Chat System - PROFILED",
    page_icon="rocket",
    layout="wide"
)

CONFIG = {
    "connection_string": os.getenv("SUPABASE_CONNECTION_STRING"),
    "table_name": "documents",
    "embed_model": "mxbai-embed-large",
    "chat_model": "llama3.2:3b",
    "embed_dim": 1024,
    "ollama_url": "http://localhost:11434"
}

@st.cache_resource
def initialize_rag_system():
    try:
        init_start = time.time()
        logger.info("Starting RAG system initialization")
        
        if not CONFIG["connection_string"]:
            st.error("SUPABASE_CONNECTION_STRING not found!")
            st.stop()

        with st.spinner("Initializing PROFILED RAG system..."):
            vector_store = SupabaseVectorStore(
                postgres_connection_string=CONFIG["connection_string"],
                collection_name=CONFIG["table_name"],
                dimension=CONFIG["embed_dim"],
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            embed_model = OllamaEmbedding(
                model_name=CONFIG["embed_model"], 
                base_url=CONFIG["ollama_url"]
            )
            
            llm = Ollama(
                model=CONFIG["chat_model"],
                base_url=CONFIG["ollama_url"],
                temperature=0.1,
                request_timeout=180.0,
                context_window=1024,
                num_predict=128
            )
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model
            )
            
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=8,
                embed_model=embed_model
            )
            
            similarity_filter = SimilarityPostprocessor(similarity_cutoff=0.3)
            
            chat_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[similarity_filter],
                llm=llm,
                response_mode="compact",
                streaming=False,
                verbose=True
            )
            
            total_init_time = time.time() - init_start
            logger.info(f"Total initialization: {total_init_time:.2f}s")
            
            return chat_engine, index, retriever
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

def main():
    st.title("Fast RAG Chat System - PROFILED")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask me about your documents.", "sources": []}
        ]
    
    chat_engine, index, retriever = initialize_rag_system()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask me about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            total_start = time.time()
            logger.info(f"Processing: {prompt}")
            
            stage_placeholder = st.empty()
            
            try:
                stage_placeholder.write("Stage 1: Preprocessing...")
                stage1_start = time.time()
                
                stage_placeholder.write("Stage 2: Embedding generation...")
                stage2_start = time.time()
                
                stage_placeholder.write("Stage 3: Vector search...")
                stage3_start = time.time()
                
                stage_placeholder.write("Stage 4: LLM processing...")
                stage4_start = time.time()
                
                response = chat_engine.query(prompt)
                
                stage4_end = time.time()
                total_end = time.time()
                
                stage_placeholder.empty()
                
                stage1_time = stage2_start - stage1_start
                stage2_time = stage3_start - stage2_start
                stage3_time = stage4_start - stage3_start
                stage4_time = stage4_end - stage4_start
                total_time = total_end - total_start
                
                logger.info(f"Total: {total_time:.2f}s")
                logger.info(f"Stage 1: {stage1_time:.2f}s")
                logger.info(f"Stage 2: {stage2_time:.2f}s")
                logger.info(f"Stage 3: {stage3_time:.2f}s")
                logger.info(f"Stage 4: {stage4_time:.2f}s")
                
                st.write(response.response)
                
                sources = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for node in response.source_nodes[:2]:
                        sources.append({
                            "text": node.get_content(),
                            "metadata": node.metadata,
                            "score": getattr(node, 'score', 0.0)
                        })
                
                with st.expander("PERFORMANCE BREAKDOWN", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Time", f"{total_time:.1f}s")
                        st.metric("Preprocessing", f"{stage1_time:.1f}s")
                        st.metric("Embedding", f"{stage2_time:.1f}s")
                    with col2:
                        st.metric("Vector Search", f"{stage3_time:.1f}s")
                        st.metric("LLM Processing", f"{stage4_time:.1f}s")
                        st.metric("Sources", len(sources))
                
                if sources:
                    with st.expander(f"Sources ({len(sources)} documents)"):
                        for i, source in enumerate(sources, 1):
                            file_name = source.get("metadata", {}).get("file_name", "Unknown")
                            similarity = source.get("score", 0)
                            text = source.get("text", "")[:300]
                            
                            st.markdown(f"**Source {i}: {file_name}** (Similarity: {similarity:.3f})")
                            st.markdown(f"```\n{text}...\n```")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.response,
                    "sources": sources
                })
                
            except Exception as e:
                stage_placeholder.empty()
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
