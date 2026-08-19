import os
import fnmatch
from typing import List, Optional, Tuple
from .registry import SkillNamespaceRegistry, SkillDefinition

class SkillsRouter:
    """
    skills.ms Routing Engine: Resolves documents and queries to target vector index namespaces.
    """
    def __init__(self, registry: Optional[SkillNamespaceRegistry] = None, default_namespace: str = "general_home"):
        self.registry = registry or SkillNamespaceRegistry()
        self.default_namespace = default_namespace

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
        3. Content keyword match scoring
        """
        # 1. Direct match on declared domain if provided, or return as new namespace
        if declared_domain and declared_domain.strip():
            clean_domain = declared_domain.strip()
            for skill in self.registry.list_skills():
                if (clean_domain.lower() in skill.skill_id.lower() or 
                    clean_domain.lower() in skill.namespace.lower() or
                    clean_domain.lower() in skill.name.lower()):
                    return skill.namespace

            # If it's a brand-new declared domain, normalize and return as new target namespace
            normalized_ns = clean_domain.lower().replace(" ", "_")
            return normalized_ns

        filename = os.path.basename(file_path).lower()
        filepath_lower = file_path.lower()

        # 2. File path / directory matching
        for skill in self.registry.list_skills():
            for pattern in skill.file_patterns:
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(filepath_lower, pattern):
                    return skill.namespace
            # Also check if directory name matches namespace or skill_id
            if skill.namespace in filepath_lower or skill.skill_id in filepath_lower:
                return skill.namespace

        # 3. Content scoring
        scores = self._score_text_against_skills(content[:2000])  # Sample first 2k chars
        if scores and scores[0][1] > 0:
            return scores[0][0].namespace

        return self.default_namespace

    def resolve_namespaces_for_query(self, query: str, top_k: int = 1) -> List[str]:
        """
        Determines the target vector index namespace(s) for a search query.
        """
        scores = self._score_text_against_skills(query)
        valid_matches = [skill.namespace for skill, score in scores if score > 0]

        if not valid_matches:
            return [self.default_namespace]
        
        return valid_matches[:top_k]

    def _score_text_against_skills(self, text: str) -> List[Tuple[SkillDefinition, int]]:
        text_lower = text.lower()
        high_intent_keywords = {
            "recipes_culinary": ["recipe", "cook", "bake", "fry", "boil", "dish", "ingredient"],
            "appliances_troubleshooting": ["manual", "error", "troubleshoot", "repair", "reset", "fault", "voltage"],
            "home_decor_design": ["decor", "layout", "aesthetic", "cabinetry", "interior"],
            "cleaning_maintenance": ["clean", "stain", "sanitize", "wash", "scrub", "seal", "remove", "maintenance"]
        }

        scores = []
        for skill in self.registry.list_skills():
            score = 0
            # Check high intent action words (weight 3)
            intent_words = high_intent_keywords.get(skill.namespace, [])
            for word in intent_words:
                if word in text_lower:
                    score += 3

            # Check general keywords (weight 1)
            for keyword in skill.keywords:
                if keyword in text_lower:
                    score += 1
            scores.append((skill, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
