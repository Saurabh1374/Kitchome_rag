"""
Root entrypoint for running the Kitchome RAG Streamlit application.
Usage:
    streamlit run streamlit_app.py
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ui.app import *
