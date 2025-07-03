import re
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter


class FastChunkingEvaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def sliding_window_chunk(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def semantic_chunk(self, text: str, similarity_threshold: float = 0.7, max_tokens: int = 200) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return [text]

        embeddings = self.model.encode(sentences, convert_to_tensor=True, batch_size=32)
        chunks = []
        current_chunk = []
        current_length = 0

        for i in range(len(sentences)):
            current_chunk.append(sentences[i])
            current_length += len(sentences[i].split())
            should_end = current_length >= max_tokens

            if i < len(sentences) - 1:
                sim = torch.nn.functional.cosine_similarity(embeddings[i], embeddings[i + 1], dim=0).item()
                if sim < similarity_threshold:
                    should_end = True

            if should_end or i == len(sentences) - 1:
                chunk_text = " ".join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0

        return chunks

    def recursive_chunk(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        return splitter.split_text(text)

    def evaluate_chunks(self, chunks: List[str], questions: List[Dict], k: int = 5) -> Dict:
        if not chunks:
            return self._empty_metrics()

        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True, batch_size=32)

        rr_scores = []
        ndcg_scores = []
        precision_scores = []
        context_scores = []
        faithfulness_scores = []

        hits_at_1 = hits_at_3 = hits_at_5 = 0

        for qa in questions:
            question = qa["question"]
            true_snippet = qa["answer_snippet"]
            question_embedding = self.model.encode([question], convert_to_tensor=True)
            similarities = cosine_similarity(question_embedding.cpu().numpy(), chunk_embeddings.cpu().numpy())[0]
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_chunks = [chunks[i] for i in top_k_indices]
            top_k_scores = [similarities[i] for i in top_k_indices]

            query_metrics = self._calculate_query_metrics(question, true_snippet, top_k_chunks, top_k_scores)

            rank = query_metrics["rank"]
            if rank == 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 5:
                hits_at_5 += 1

            rr_scores.append(query_metrics["rr"])
            ndcg_scores.append(query_metrics["ndcg"])
            precision_scores.append(query_metrics["precision"])
            context_scores.append(query_metrics["context_relevance"])
            faithfulness_scores.append(query_metrics["faithfulness"])

        total = len(questions)
        return {
            "recall_at_1": hits_at_1 / total,
            "recall_at_3": hits_at_3 / total,
            "recall_at_5": hits_at_5 / total,
            "mrr": np.mean(rr_scores),
            "ndcg": np.mean(ndcg_scores),
            "precision_at_k": np.mean(precision_scores),
            "context_relevance": np.mean(context_scores),
            "faithfulness": np.mean(faithfulness_scores),
            "hit_rate": hits_at_5 / total,
            "num_chunks": len(chunks),
            "avg_chunk_length": np.mean([len(c.split()) for c in chunks])
        }

    def _calculate_query_metrics(self, question, true_snippet, retrieved_chunks, scores):
        relevant_positions = []
        relevance_scores = []

        for i, chunk in enumerate(retrieved_chunks):
            relevance = self._calculate_relevance(true_snippet, chunk)
            if relevance > 0.5:
                relevant_positions.append(i + 1)
                relevance_scores.append(relevance)

        rank = relevant_positions[0] if relevant_positions else len(retrieved_chunks) + 1
        rr = 1.0 / rank if relevant_positions else 0
        ndcg = self._calculate_ndcg(relevance_scores, relevant_positions) if relevance_scores else 0
        precision = len(relevant_positions) / len(retrieved_chunks)
        context_relevance = self._calculate_context_relevance(question, retrieved_chunks)
        faithfulness = self._calculate_faithfulness(true_snippet, retrieved_chunks)

        return {
            "rank": rank, "rr": rr, "ndcg": ndcg,
            "precision": precision,
            "context_relevance": context_relevance,
            "faithfulness": faithfulness
        }

    def _calculate_relevance(self, true_snippet: str, chunk: str) -> float:
        if true_snippet.lower() in chunk.lower():
            return 1.0

        embeddings = self.model.encode([true_snippet, chunk], convert_to_tensor=True)
        semantic_sim = cosine_similarity(embeddings[0:1].cpu().numpy(), embeddings[1:2].cpu().numpy())[0][0]

        true_words = set(true_snippet.lower().split())
        chunk_words = set(chunk.lower().split())
        word_overlap = len(true_words.intersection(chunk_words)) / len(true_words) if true_words else 0

        return 0.7 * semantic_sim + 0.3 * word_overlap

    def _calculate_ndcg(self, relevance_scores: List[float], positions: List[int]) -> float:
        dcg = sum(score / np.log2(pos + 1) for score, pos in zip(relevance_scores, positions))
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(sorted(relevance_scores, reverse=True)))
        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_context_relevance(self, question: str, chunks: List[str]) -> float:
        embeddings = self.model.encode([question] + chunks, convert_to_tensor=True)
        similarities = cosine_similarity(embeddings[0:1].cpu().numpy(), embeddings[1:].cpu().numpy())[0]
        return np.mean(similarities)

    def _calculate_faithfulness(self, true_snippet: str, chunks: List[str]) -> float:
        embeddings = self.model.encode([true_snippet] + chunks, convert_to_tensor=True)
        similarities = cosine_similarity(embeddings[0:1].cpu().numpy(), embeddings[1:].cpu().numpy())[0]
        return np.max(similarities)

    def _empty_metrics(self) -> Dict:
        return {
            "recall_at_1": 0, "recall_at_3": 0, "recall_at_5": 0,
            "mrr": 0, "ndcg": 0, "precision_at_k": 0,
            "context_relevance": 0, "faithfulness": 0, "hit_rate": 0,
            "num_chunks": 0, "avg_chunk_length": 0
        }


def run_fast_evaluation(pdf_path: str, qna_path: str) -> pd.DataFrame:
    reader = PdfReader(pdf_path)
    all_text = " ".join([p.extract_text() or "" for p in reader.pages])

    with open(qna_path) as f:
        qna_data = json.load(f)

    evaluator = FastChunkingEvaluator()

    strategies = [
        ("sliding_100_20", lambda text: evaluator.sliding_window_chunk(text, 100, 20)),
        ("sliding_150_30", lambda text: evaluator.sliding_window_chunk(text, 150, 30)),
        ("sliding_200_40", lambda text: evaluator.sliding_window_chunk(text, 200, 40)),
        ("sliding_250_50", lambda text: evaluator.sliding_window_chunk(text, 250, 50)),
        ("semantic_065_150", lambda text: evaluator.semantic_chunk(text, 0.65, 150)),
        ("semantic_070_200", lambda text: evaluator.semantic_chunk(text, 0.70, 200)),
        ("semantic_075_250", lambda text: evaluator.semantic_chunk(text, 0.75, 250)),
        ("recursive_800_100", lambda text: evaluator.recursive_chunk(text, 800, 100)),
        ("recursive_1000_150", lambda text: evaluator.recursive_chunk(text, 1000, 150)),
        ("recursive_1200_200", lambda text: evaluator.recursive_chunk(text, 1200, 200)),
    ]

    results = []

    for strategy_name, chunker_func in strategies:
        print(f"Evaluating strategy: {strategy_name}")
        start_time = time.time()
        try:
            chunks = chunker_func(all_text)
            if not chunks:
                print(f"Skipped {strategy_name} (no chunks generated).")
                continue

            metrics = evaluator.evaluate_chunks(chunks, qna_data)
            if "mrr" not in metrics:
                print(f"Skipped {strategy_name} (incomplete metrics).")
                continue

            metrics.update({"strategy": strategy_name, "processing_time": time.time() - start_time})
            results.append(metrics)
        except Exception as e:
            print(f"Error in {strategy_name}: {e}")

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("No valid results generated. Check your PDF or QnA input.")

    df['combined_score'] = (df['mrr'] + df['ndcg'] + df['recall_at_1']) / 3
    df.sort_values("combined_score", ascending=False, inplace=True)

    df.to_csv("fast_chunking_evaluation.csv", index=False)
    create_evaluation_plots(df)

    with open("best_chunking_strategy.json", "w") as f:
        json.dump(df.iloc[0].to_dict(), f, indent=2)

    return df


def create_evaluation_plots(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Chunking Strategy Evaluation Results", fontsize=16)

    metrics = ['mrr', 'ndcg', 'recall_at_1', 'recall_at_5']
    x = np.arange(len(df))
    width = 0.2

    for i, metric in enumerate(metrics):
        axes[0, 0].bar(x + i * width, df[metric], width, label=metric.upper())

    axes[0, 0].set_title("Performance Metrics Comparison")
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(df['strategy'], rotation=45, ha='right')
    axes[0, 0].legend()

    axes[0, 1].scatter(df['processing_time'], df['mrr'], s=df['num_chunks'] / 10,
                       c=df['avg_chunk_length'], cmap='viridis')
    axes[0, 1].set_title("Processing Time vs MRR")

    axes[1, 0].scatter(df['num_chunks'], df['avg_chunk_length'], c=df['mrr'], cmap='plasma', s=100)
    axes[1, 0].set_title("Chunk Characteristics")

    axes[1, 1].bar(range(len(df)), df['combined_score'], color='skyblue')
    axes[1, 1].set_title("Combined Performance Score")
    axes[1, 1].set_xticks(range(len(df)))
    axes[1, 1].set_xticklabels(df['strategy'], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("chunking_evaluation_plots.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    PDF_PATH = "medicare.pdf"
    QNA_PATH = "eval_qna.json"
    results_df = run_fast_evaluation(PDF_PATH, QNA_PATH)