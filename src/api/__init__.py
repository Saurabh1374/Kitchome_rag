"""
Kitchome RAG API Package.
Exposes REST endpoints for querying, strict single-document ingestion,
user onboarding, and administrative access control.
"""

from .main import app

__all__ = ["app"]
