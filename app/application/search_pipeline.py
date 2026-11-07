"""Hybrid retrieval with BM25, reciprocal rank fusion and bounded context."""
import math
import re
from collections import Counter
from dataclasses import replace

from app.application.retrieval import build_context
from app.domain.retrieval import RetrievalResult


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def bm25(question, corpus):
    """Robertson BM25, k1=1.5, b=.75; only positive matches are returned."""
    if not corpus:
        return []
    terms = [Counter(tokens(item.text)) for item in corpus]
    lengths = [sum(term.values()) for term in terms]
    average = sum(lengths) / len(lengths) or 1
    query = set(tokens(question))
    frequency = {word: sum(word in term for term in terms) for word in query}
    scored = []
    for item, term, length in zip(corpus, terms, lengths):
        score = 0.0
        for word in query:
            count = term[word]
            idf = math.log(1 + (len(corpus) - frequency[word] + .5) / (frequency[word] + .5))
            score += idf * count * 2.5 / (count + 1.5 * (.25 + .75 * length / average))
        if score > 0:
            scored.append((score, item))
    return [item for _, item in sorted(scored, key=lambda pair: (-pair[0], str(pair[1].document_id), pair[1].chunk_index))]


def fuse(*rankings):
    scores, matches = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            key = (item.document_id, item.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            matches.setdefault(key, item)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    # Keep vector cosine similarity separate from fusion scores.
    return [matches[key] for key in sorted(scores, key=lambda k: (-scores[k], str(k[0]), k[1]))]


class CrossEncoderReranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, question, matches):
        if not matches:
            return []
        scores = self.model.predict([(question, item.text) for item in matches])
        return [item for _, item in sorted(zip(scores, matches), key=lambda pair: -float(pair[0]))]


class SearchPipeline:
    def __init__(self, vector, repository, *, hybrid=False, reranker=None,
                 multi_query=False, neighbors=False, context_chars=12000):
        if context_chars < 1:
            raise ValueError('context_chars must be positive')
        self.vector = vector
        self.repository = repository
        self.hybrid = hybrid
        self.reranker = reranker
        self.multi_query = multi_query
        self.neighbors = neighbors
        self.context_chars = context_chars

    def retrieve(self, question, top_k=5, document_id=None, min_similarity=None):
        if not 1 <= top_k <= 50:
            raise ValueError('top_k must be between 1 and 50')
        result = self.vector.retrieve(question, min(50, top_k * 4), document_id, min_similarity)
        rankings = [list(result.matches)]
        if self.multi_query:
            # Deterministic query decomposition; no extra LLM calls.
            parts = re.split(r'\s+(?:and|also)\s+|[?;]', question)
            for part in [p.strip() for p in parts if p.strip()][:3]:
                if part != question.strip():
                    rankings.append(list(self.vector.retrieve(part, min(50, top_k * 4), document_id, min_similarity).matches))
        corpus = None
        if self.hybrid:
            corpus = self.repository.corpus(self.vector.model_name, document_id)
            lexical = bm25(question, corpus)[:50]
            if lexical:
                # Return real cosine scores, also for lexical-only candidates.
                from app.application.embeddings import embed_query
                query = embed_query(question, self.vector.embedding_provider)
                lexical = self.repository.score_candidates(lexical, query)
                if min_similarity is not None:
                    lexical = [m for m in lexical if m.similarity >= min_similarity]
            rankings.append(lexical)
        matches = fuse(*rankings)
        if self.reranker:
            matches = self.reranker.rerank(question, matches)
        matches = matches[:top_k]
        if self.neighbors and matches:
            matches = self.repository.expand_neighbors(matches, self.vector.model_name)
        bounded = []
        for item in matches:
            candidate = item
            if len(build_context([*bounded, candidate])) > self.context_chars:
                # Preserve source metadata and trim only evidence text.
                remaining = self.context_chars - len(build_context([*bounded, replace(item, text='')]))
                if remaining > 0:
                    bounded.append(replace(item, text=item.text[:remaining]))
                break
            bounded.append(candidate)
        return RetrievalResult(result.question, tuple(bounded), build_context(bounded))
