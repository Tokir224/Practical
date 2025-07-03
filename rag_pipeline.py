import os
import re
import json
import torch
import asyncio
import warnings
from typing import List, Dict, Optional
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document, BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


warnings.filterwarnings('ignore')
load_dotenv(override=True)


class HybridRetriever(BaseRetriever):
    def __init__(self, vectorstore, documents: List[Document], k=4):
        """
        Initialize a hybrid retriever that combines vector search and BM25.
        """
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k
        self._documents = documents

        # Initialize BM25 retriever with documents
        self._bm25_retriever = BM25Retriever.from_documents(
            self._documents,
            k=self._k
        )

    @property
    def vectorstore(self):
        return self._vectorstore

    @property
    def k(self):
        return self._k

    @property
    def bm25_retriever(self):
        return self._bm25_retriever

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Create ensemble retriever that combines both approaches
        ensemble = EnsembleRetriever(
            retrievers=[
                self.vectorstore.as_retriever(
                    search_type="mmr",  # Use MMR for better diversity
                    search_kwargs={
                        "k": self.k,
                        "fetch_k": self.k * 2,  # Fetch more candidates for MMR
                        "lambda_mult": 0.5  # Balance between relevance and diversity
                    }
                ),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]  # Favor semantic search slightly
        )

        # Get combined results
        results = ensemble.get_relevant_documents(query)
        return results[:self.k]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_relevant_documents, query)


class OptimizedRAGSystem:
    def __init__(
        self,
        best_strategy_path: str = "best_chunking_strategy.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama3-70b-8192"
    ):
        # Load best strategy
        with open(best_strategy_path, 'r') as f:
            self.best_config = json.load(f)
        print(f"Using best strategy: {self.best_config['strategy']}")

        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device}
        )

        # Initialize LLM (ChatGroq)
        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=groq_model,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            self.llm = None
            raise ValueError("No Groq API key provided. Please set the GROQ_API_KEY environment variable.")

        self.documents = []
        self.vector_store = None
        self.hybrid_retriever = None
        self.qa_chain = None

    def create_optimal_chunks(self, text: str, page_numbers: List[int]) -> List[Document]:
        strategy = self.best_config['strategy']
        documents = []
        current_pos = 0

        # Select chunking strategy
        if strategy.startswith('recursive'):
            parts = strategy.split('_')
            chunk_size = int(parts[1])
            overlap = int(parts[2])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            chunks = splitter.split_text(text)

        elif strategy.startswith('sliding'):
            parts = strategy.split('_')
            chunk_size = int(parts[1])
            overlap = int(parts[2])
            words = text.split()
            chunks = []

            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)
                if i + chunk_size >= len(words):
                    break

        elif strategy.startswith('semantic'):
            parts = strategy.split('_')
            similarity_threshold = float("0." + parts[1])
            max_tokens = int(parts[2])

            # Lazy load model only if needed
            if not hasattr(self, '_semantic_model'):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) < 2:
                chunks = [text]
            else:
                embeddings = self._semantic_model.encode(sentences, convert_to_tensor=True, batch_size=32)
                chunks = []
                current_chunk = []
                current_length = 0

                for i in range(len(sentences)):
                    sent = sentences[i]
                    current_chunk.append(sent)
                    current_length += len(sent.split())

                    should_end = current_length >= max_tokens
                    if i < len(sentences) - 1:
                        sim = torch.nn.functional.cosine_similarity(
                            embeddings[i], embeddings[i + 1], dim=0
                        ).item()
                        if sim < similarity_threshold:
                            should_end = True

                    if should_end or i == len(sentences) - 1:
                        chunk_text = " ".join(current_chunk).strip()
                        if chunk_text:
                            chunks.append(chunk_text)
                        current_chunk = []
                        current_length = 0

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        # Convert chunks to LangChain Document objects
        chars_per_page = len(text) / len(page_numbers) if page_numbers else 1000

        for chunk in chunks:
            chunk_start = text.find(chunk, current_pos)
            if chunk_start == -1:
                chunk_start = current_pos  # fallback
            estimated_page = min(int(chunk_start / chars_per_page), len(page_numbers) - 1)
            page_num = page_numbers[estimated_page] if page_numbers else 1

            doc = Document(
                page_content=chunk,
                metadata={
                    'page': page_num,
                    'chunk_id': len(documents),
                    'length': len(chunk),
                    'start_pos': chunk_start
                }
            )
            documents.append(doc)
            current_pos = chunk_start + len(chunk)

        return documents

    def build_vector_store(self, pdf_path: str):
        reader = PdfReader(pdf_path)
        all_text = ""
        page_numbers = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                all_text += text + " "
                page_numbers.append(page_num)

        self.documents = self.create_optimal_chunks(all_text, page_numbers)
        self.vector_store = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embedding_model
        )

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vectorstore=self.vector_store,
            documents=self.documents,
            k=5
        )

        # Initialize QA chain
        self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        """Initialize QA chain with custom prompt."""
        qa_prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context:
            {context}

            Question: {question}""",
            input_variables=["context", "question"]
        )

        # Single QA chain using stuff method
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=qa_prompt
        )

    def query(self, question: str, k: int = 5) -> Dict:
        """
        Main query method using ensemble retrieval with QA chain.

        Args:
            question: The question to ask
            k: Number of relevant chunks to retrieve

        Returns:
            Dict containing answer, source_page, chunk_size, and confidence_score
        """
        if self.hybrid_retriever is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")

        try:
            # Get relevant documents using ensemble retrieval
            docs = self.hybrid_retriever.get_relevant_documents(question)

            if not docs:
                return self._no_answer_response()

            # Use QA chain to generate answer
            result = self.qa_chain(
                {"input_documents": docs, "question": question}
            )

            # Get the best document (first one from ensemble)
            best_doc = docs[0]

            # Calculate dynamic confidence score
            confidence_score = self._calculate_confidence_score(question, docs, result["output_text"])

            return {
                "answer": result["output_text"],
                "source_page": best_doc.metadata.get("page", 0),
                "chunk_size": best_doc.metadata.get("length", len(best_doc.page_content)),
                "confidence_score": confidence_score
            }

        except Exception as e:
            print(f"Query failed: {e}")
            return self._no_answer_response()

    def ask(self, question: str) -> str:
        """Simple ask method that returns just the answer."""
        result = self.query(question)
        return result.get('answer', 'No answer found.')

    @staticmethod
    def _no_answer_response() -> Dict:
        return {
            'answer': "No relevant information found to answer the question.",
            'source_page': 0,
            'chunk_size': 0,
            'confidence_score': 0.0
        }

    def save_vector_store(self, path: str):
        """Save the vector store to disk."""
        if self.vector_store is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")

        self.vector_store.save_local(path)

        # Also save the documents and configuration
        with open(f"{path}_documents.json", 'w') as f:
            json.dump([
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in self.documents
            ], f, indent=2)

        print(f"Vector store saved to {path}")

    def load_vector_store(self, path: str):
        """Load the vector store from disk."""
        self.vector_store = FAISS.load_local(path, self.embedding_model)

        # Load documents
        with open(f"{path}_documents.json", 'r') as f:
            doc_data = json.load(f)

        self.documents = [
            Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            )
            for doc in doc_data
        ]

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vectorstore=self.vector_store,
            documents=self.documents,
            k=5
        )

        # Initialize QA chain
        self._initialize_qa_chain()

        print(f"Vector store loaded from {path}")

    def _calculate_confidence_score(self, question: str, docs: List[Document], answer: str) -> float:
        """
        Calculate dynamic confidence score based on three key factors.

        Args:
            question: The original question
            docs: Retrieved documents
            answer: Generated answer

        Returns:
            Float confidence score between 0.0 and 1.0
        """
        if not docs or not answer:
            return 0.0

        # Factor 1: Retrieval quality (semantic similarity between question and docs)
        retrieval_score = self._calculate_retrieval_score(question, docs)

        # Factor 2: Answer quality (completeness and relevance of the answer)
        answer_score = self._calculate_answer_score(question, answer, docs)

        # Factor 3: Document diversity (variety in retrieved content)
        diversity_score = self._calculate_diversity_score(docs)

        # Weighted combination: retrieval and answer are most important
        confidence = (
            retrieval_score * 0.45 +
            answer_score * 0.45 +
            diversity_score * 0.10
        )

        return max(0.0, min(1.0, confidence))

    def _calculate_retrieval_score(self, question: str, docs: List[Document]) -> float:
        """Calculate score based on semantic similarity between question and retrieved docs."""
        if not docs:
            return 0.0

        try:
            # Get embedding for question
            question_embedding = self.embedding_model.embed_query(question)

            # Get embeddings for top 3 documents
            doc_texts = [doc.page_content[:400] for doc in docs[:3]]
            doc_embeddings = self.embedding_model.embed_documents(doc_texts)

            # Calculate cosine similarities
            similarities = []
            for doc_emb in doc_embeddings:
                similarity = sum(a * b for a, b in zip(question_embedding, doc_emb))
                similarities.append(similarity)

            # Weighted average (prioritize top document)
            if len(similarities) >= 3:
                weighted_sim = (similarities[0] * 0.6 + similarities[1] * 0.3 + similarities[2] * 0.1)
            elif len(similarities) == 2:
                weighted_sim = (similarities[0] * 0.7 + similarities[1] * 0.3)
            else:
                weighted_sim = similarities[0]

            # Normalize to 0-1 range
            return max(0.0, (weighted_sim + 1) / 2)

        except Exception as e:
            print(f"Error calculating retrieval score: {e}")
            return 0.5

    def _calculate_answer_score(self, question: str, answer: str, docs: List[Document]) -> float:
        """Calculate score based on answer quality and relevance."""
        if not answer or answer.lower().strip() in ["no answer found.", "i don't know", "no relevant information found"]:
            return 0.1

        score = 0.3  # Base score

        # Check answer length (not too short, not too long)
        answer_length = len(answer.split())
        if 10 <= answer_length <= 150:
            score += 0.3
        elif 5 <= answer_length <= 200:
            score += 0.2

        # Check content overlap with source documents
        doc_content = " ".join([doc.page_content for doc in docs[:2]]).lower()
        answer_words = set(answer.lower().split())
        doc_words = set(doc_content.split())

        if answer_words:
            overlap_ratio = len(answer_words.intersection(doc_words)) / len(answer_words)
            score += min(0.3, overlap_ratio)

        # Penalty for uncertainty phrases
        uncertainty_phrases = ["i don't know", "not sure", "cannot determine", "unclear"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            score -= 0.2

        # Bonus for structured answers
        if len([s for s in answer.split('.') if s.strip()]) >= 2:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _calculate_diversity_score(self, docs: List[Document]) -> float:
        """Calculate score based on diversity of retrieved documents."""
        if len(docs) < 2:
            return 0.6  # Default score for single document

        try:
            # Use first 200 chars of each document for efficiency
            doc_texts = [doc.page_content[:200] for doc in docs[:4]]
            doc_embeddings = self.embedding_model.embed_documents(doc_texts)

            # Calculate pairwise similarities
            similarities = []
            for i in range(len(doc_embeddings)):
                for j in range(i + 1, len(doc_embeddings)):
                    sim = sum(a * b for a, b in zip(doc_embeddings[i], doc_embeddings[j]))
                    similarities.append(sim)

            if not similarities:
                return 0.6

            # Average similarity
            avg_similarity = sum(similarities) / len(similarities)

            # Convert to diversity score (lower similarity = higher diversity)
            diversity = 1 - max(0, (avg_similarity + 1) / 2)

            # Ensure reasonable range
            return max(0.3, min(0.9, diversity))

        except Exception as e:
            print(f"Error calculating diversity score: {e}")
            return 0.6


def main():
    """Demo with simplified Q&A."""
    rag = OptimizedRAGSystem(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model="llama3-70b-8192"
    )

    # Build vector store
    rag.build_vector_store("medicare.pdf")

    demo_queries = [
        "What are the eligibility criteria for Medicare Advantage Plans?",
        "How do I enroll in Medicare Part B?",
        "What is the difference between Medicare Part A and Part B?",
        "What are the important deadlines for Medicare enrollment?",
        "Can I change my Medicare plan during the year?"
    ]

    print("=== Testing Ensemble Q&A ===")
    for query in demo_queries:
        result = rag.query(query)
        print(f"Question: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Source Page: {result['source_page']}")
        print(f"Chunk Size: {result['chunk_size']}")
        print(f"Confidence: {result['confidence_score']}")
        print("-" * 60)

    print("\n=== Testing Simple Ask Method ===")
    for query in demo_queries[:2]:
        answer = rag.ask(query)
        print(f"Question: {query}")
        print(f"Answer: {answer}")
        print("-" * 40)


if __name__ == "__main__":
    main()
