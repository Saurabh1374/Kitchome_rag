import os
import re
import fnmatch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from .registry import SkillNamespaceRegistry, SkillDefinition

if TYPE_CHECKING:
    from ..ingestion.embedder import EmbeddingEngine

class SkillsRouter:
    """
    skills.ms Ensemble Routing Engine:
    Resolves documents and user queries to target vector index namespaces using a two-channel
    ensemble (Lexical Token-Boundary + Semantic Prototype) with confidence margin arbitration.
    """
    def __init__(
        self, 
        registry: Optional[SkillNamespaceRegistry] = None, 
        default_namespace: str = "general_home",
        embedder: Optional[Any] = None,
        alpha: float = 0.65,
        margin_threshold: float = 0.18
    ):
        self.registry = registry or SkillNamespaceRegistry()
        self.default_namespace = default_namespace
        if embedder is None:
            from ..ingestion.embedder import EmbeddingEngine
            self.embedder = EmbeddingEngine()
        else:
            self.embedder = embedder
        self.alpha = alpha  # Weight for semantic prototype score vs lexical score
        self.margin_threshold = margin_threshold  # Confidence margin threshold Δ = S1 - S2
        
        # Cache pre-computed exemplar vectors and centroid representations
        self._exemplar_embeddings: Dict[str, List[List[float]]] = {}
        self._exemplar_centroids: Dict[str, List[float]] = {}
        self._build_exemplar_index()

    def _build_exemplar_index(self) -> None:
        """Pre-computes and caches normalized embedding vectors for each skill's exemplars."""
        for skill in self.registry.list_skills():
            if skill.exemplars:
                vectors = [self.embedder.embed_text(ex) for ex in skill.exemplars]
                self._exemplar_embeddings[skill.namespace] = vectors
                
                # Compute centroid vector and normalize
                arr = np.array(vectors, dtype=np.float32)
                centroid = np.mean(arr, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                self._exemplar_centroids[skill.namespace] = centroid.tolist()

    def resolve_namespace_for_document(
        self, 
        file_path: str, 
        content: str, 
        declared_domain: Optional[str] = None
    ) -> str:
        """
        Determines the target vector index namespace for a document based on:
        1. Explicit declared domain tag
        2. File path pattern matching
        3. Content ensemble scoring
        """
        # 1. Direct match on declared domain if provided, or return as new namespace
        if declared_domain and declared_domain.strip():
            clean_domain = declared_domain.strip()
            for skill in self.registry.list_skills():
                if (clean_domain.lower() in skill.skill_id.lower() or 
                    clean_domain.lower() in skill.namespace.lower() or
                    clean_domain.lower() in skill.name.lower()):
                    return skill.namespace

            normalized_ns = clean_domain.lower().replace(" ", "_")
            return normalized_ns

        filename = os.path.basename(file_path).lower()
        filepath_lower = file_path.lower()

        # 2. File path / directory matching
        for skill in self.registry.list_skills():
            for pattern in skill.file_patterns:
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(filepath_lower, pattern):
                    return skill.namespace
            if skill.namespace in filepath_lower or skill.skill_id in filepath_lower:
                return skill.namespace

        # 3. Content ensemble scoring on document sample
        sample_text = content[:2500]
        decision = self.resolve_namespaces_with_confidence(sample_text)
        if decision["namespaces"]:
            return decision["namespaces"][0]

        return self.default_namespace

    def resolve_namespaces_for_query(self, query: str, top_k: int = 1) -> List[str]:
        """
        Determines the target vector index namespace(s) for a search query.
        Backward-compatible signature that returns top matching namespace(s).
        """
        decision = self.resolve_namespaces_with_confidence(query, top_k=top_k)
        return decision["namespaces"]

    def resolve_namespaces_with_confidence(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Two-Channel Ensemble Resolution with Margin Arbitration:
        - Computes Lexical Score (S_lex) with token-boundary regex & mutual-exclusion damping.
        - Computes Semantic Prototype Score (S_sem) via cosine similarity with exemplars.
        - Fuses: S_final = α·S_sem + (1-α)·S_lex
        - Margin Arbitration: Δ = S_1 - S_2. If Δ >= τ, single namespace; else top-2.
        """
        skills = self.registry.list_skills()
        if not skills:
            return {
                "namespaces": [self.default_namespace],
                "confidence_margin": 1.0,
                "top_score": 0.0,
                "fused_scores": {}
            }

        lex_scores = self._score_lexical(query, skills)
        sem_scores = self._score_semantic(query, skills)

        # Score fusion
        fused_scores: Dict[str, float] = {}
        for skill in skills:
            ns = skill.namespace
            s_lex = lex_scores.get(ns, 0.0)
            s_sem = sem_scores.get(ns, 0.0)
            fused_scores[ns] = (self.alpha * s_sem) + ((1.0 - self.alpha) * s_lex)

        # Sort namespaces by fused score descending
        sorted_skills = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_ns, top_score = sorted_skills[0]
        second_ns, second_score = sorted_skills[1] if len(sorted_skills) > 1 else ("", 0.0)

        confidence_margin = max(0.0, top_score - second_score)

        # Decision rule:
        # If top score is very weak, fallback to default
        if top_score < 0.12:
            resolved = [self.default_namespace]
        # If high confidence margin, return single namespace
        elif confidence_margin >= self.margin_threshold or top_k == 1:
            resolved = [top_ns]
        else:
            # Low margin / ambiguous intent: return top-2 namespaces
            resolved = [top_ns]
            if second_score > 0.15 and second_ns:
                resolved.append(second_ns)

        return {
            "namespaces": resolved,
            "confidence_margin": round(confidence_margin, 4),
            "top_score": round(top_score, 4),
            "fused_scores": {k: round(v, 4) for k, v in fused_scores.items()},
            "lexical_scores": {k: round(v, 4) for k, v in lex_scores.items()},
            "semantic_scores": {k: round(v, 4) for k, v in sem_scores.items()}
        }

    def _score_lexical(self, text: str, skills: List[SkillDefinition]) -> Dict[str, float]:
        """
        Channel 1: Token-boundary matching with domain suppression to eradicate false positives.
        """
        text_lower = text.lower()
        high_intent_keywords = {
            "recipes_culinary": ["recipe", "recipes", "cook", "cooking", "bake", "baking", "fry", "boil", "sear", "ingredient", "dish", "meal", "pasta", "chicken", "salmon", "beef"],
            "appliances_troubleshooting": ["manual", "error", "troubleshoot", "repair", "reset", "fault", "voltage", "watts", "fuse", "overheating", "compressor", "leak"],
            "home_decor_design": ["decor", "layout", "aesthetic", "cabinetry", "cabinet", "interior", "countertop", "quartz", "granite", "backsplash", "island", "lighting"],
            "cleaning_maintenance": ["clean", "cleaning", "stain", "sanitize", "wash", "scrub", "seal", "remove", "maintenance", "unclog", "drain", "descaling", "bleach", "vinegar"],
            "academia_research": ["research", "paper", "methodology", "empirical", "evaluation", "architecture", "dataset", "citations", "latex", "theorem", "transformer", "benchmark"]
        }

        # Check for culinary presence
        has_culinary_context = any(
            re.search(r'\b' + re.escape(w) + r'\b', text_lower)
            for w in ["cook", "bake", "fry", "sear", "recipe", "ingredient", "salmon", "chicken", "pasta", "dish", "food"]
        )

        raw_scores: Dict[str, float] = {}
        max_possible = 0.0

        for skill in skills:
            score = 0.0
            intent_words = high_intent_keywords.get(skill.namespace, [])

            # High-intent terms (weight 3)
            for word in intent_words:
                if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    score += 3.0

            # General keywords (weight 1)
            for kw in skill.keywords:
                if kw and re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                    score += 1.5

            # Domain suppression: If strong culinary context is present, damp cleaning_maintenance
            # to prevent 'clean the chicken / prep salmon' from false-positiving into cleaning
            if skill.namespace == "cleaning_maintenance" and has_culinary_context:
                score *= 0.2

            raw_scores[skill.namespace] = score
            if score > max_possible:
                max_possible = score

        # Normalize to [0.0, 1.0]
        if max_possible > 0:
            return {k: v / max_possible for k, v in raw_scores.items()}
        return {k: 0.0 for k in raw_scores}

    def _score_semantic(self, text: str, skills: List[SkillDefinition]) -> Dict[str, float]:
        """
        Channel 2: Semantic Prototype Cosine Similarity against pre-computed exemplar vectors.
        """
        query_vec = np.array(self.embedder.embed_text(text), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return {s.namespace: 0.0 for s in skills}
        query_vec = query_vec / q_norm

        scores: Dict[str, float] = {}
        for skill in skills:
            ns = skill.namespace
            exemplar_vecs = self._exemplar_embeddings.get(ns)
            
            # Lazy indexing for dynamic skills
            if exemplar_vecs is None:
                if skill.exemplars:
                    exemplar_vecs = [self.embedder.embed_text(ex) for ex in skill.exemplars]
                    self._exemplar_embeddings[ns] = exemplar_vecs
                elif skill.keywords:
                    # Fallback to embedding keyword phrase for dynamic skills
                    exemplar_vecs = [self.embedder.embed_text(" ".join(skill.keywords))]
                    self._exemplar_embeddings[ns] = exemplar_vecs

            if not exemplar_vecs:
                scores[ns] = 0.0
                continue

            # Compute cosine similarity across all exemplars and take maximum
            ex_matrix = np.array(exemplar_vecs, dtype=np.float32)
            sims = np.dot(ex_matrix, query_vec)
            max_sim = float(np.max(sims))
            scores[ns] = max(0.0, max_sim)

        return scores
