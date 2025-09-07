import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PPLX_API_KEY'] = os.getenv("PPLX_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate

from typing import List
import re
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from sap_tables import MODULE_TABLES

o4_mini = ChatOpenAI(model='o4-mini')

import re
from typing import List
from langchain_core.documents import Document


class TableAgent:
    def __init__(self, module: str, top_k: int = 5):
        self.module = module.upper()
        if self.module not in MODULE_TABLES:
            raise ValueError(f"Module {module} not supported.")
        self.table_data = MODULE_TABLES[self.module]
        
        # Create Document objects with metadata for both retrievers
        documents = [
            Document(
                page_content=desc,
                metadata={"table": table}
            )
            for table, desc in self.table_data.items()
        ]
        
        # Embedding model
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Dense vector store (FAISS)
        self.vector_store = FAISS.from_documents(
            documents,
            self.embedding,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        
        # Sparse retriever (BM25)
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=self._tokenize
        )
        
        # Hybrid ensemble retriever (dense + sparse)
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_store.as_retriever(search_kwargs={"k": top_k}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]  # Tune as needed
        )
        
        # Cross-encoder for reranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm = o4_mini
        self.top_k = min(top_k, len(self.table_data))

    def _tokenize(self, text: str) -> List[str]:
        """Regex-based tokenizer (no NLTK dependency)"""
        return [token.lower() for token in re.findall(r'\b\w+\b', text)]

    def route(self, query: str) -> str:
        # 1. Hybrid retrieval (BM25 + FAISS)
        docs = self.hybrid_retriever.invoke(query)
        if not docs:
            return "No confident match found"

        # 2. Cross-encoder reranking
        pairs = [(query, doc.page_content) for doc in docs]
        ce_scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(docs, ce_scores), key=lambda x: x[1], reverse=True)

        # 3. Prepare candidates for LLM
        candidate_tables = [doc.metadata["table"] for doc, _ in reranked]
        candidate_scores = [score for _, score in reranked]
        candidates_info = "\n".join([
            f"{table} (relevance: {score:.2f}): {self.table_data[table]}"
            for table, score in zip(candidate_tables, candidate_scores)
        ])

        # 4. LLM selection prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are an SAP table expert. Select the MOST RELEVANT TABLES (There may be more than one) from these candidates:"
             "\n\n{candidates_info}"
             "\n\nUser query: {query}"
             "\n\nReturn ONLY the table names from the listed options. If more than one, separate by a comma."),
        ])

        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({
            "candidates_info": candidates_info,
            "query": query
        }).strip()