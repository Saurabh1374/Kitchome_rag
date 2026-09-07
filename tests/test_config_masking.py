import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from config import (
    mask_secret,
    EmbedderConfig,
    PGVectorConfig,
    AppConfig,
    config
)

def test_mask_secret_helper():
    """Verify mask_secret helper masks correctly for various inputs."""
    assert mask_secret(None) == "None"
    assert mask_secret("") == ""
    assert mask_secret("abc") == "********"
    assert mask_secret("1234") == "********"
    assert mask_secret("supersecretpass", visible_chars=4) == "supe...******"
    assert mask_secret("hf_1234567890abcdef", visible_chars=4) == "hf_1...******"
    assert mask_secret("supersecretpass", full_mask=True) == "********"

def test_embedder_config_masking():
    """Verify EmbedderConfig masks api_key in str, repr, and get_masked_dict."""
    cfg = EmbedderConfig(api_key="hf_test_secret_key_98765")

    # String representations must NEVER leak the plain secret
    assert "hf_test_secret_key_98765" not in repr(cfg)
    assert "hf_test_secret_key_98765" not in str(cfg)
    assert "hf_t...******" in repr(cfg)
    assert "hf_t...******" in str(cfg)

    # get_masked_dict must have the masked key
    masked_d = cfg.get_masked_dict()
    assert masked_d["api_key"] == "hf_t...******"

    # Actual attribute must remain unmasked for downstream API requests
    assert cfg.api_key == "hf_test_secret_key_98765"

def test_pgvector_config_masking():
    """Verify PGVectorConfig masks password and defaults to empty string."""
    cfg = PGVectorConfig(password="super_secret_db_pass")

    # String representations must NEVER leak the plain password
    assert "super_secret_db_pass" not in repr(cfg)
    assert "super_secret_db_pass" not in str(cfg)
    assert "supe...******" in repr(cfg)
    assert "supe...******" in str(cfg)

    # get_masked_dict must have the masked password
    masked_d = cfg.get_masked_dict()
    assert masked_d["password"] == "supe...******"

    # Actual attribute must remain unmasked for psycopg2 / pgvector connections
    assert cfg.password == "super_secret_db_pass"

    # Default password must not be hardcoded to "postgres"
    fresh_cfg = PGVectorConfig()
    assert fresh_cfg.password == "" or fresh_cfg.password == os.getenv("POSTGRES_PASSWORD", "")

def test_app_config_masking():
    """Verify AppConfig masks nested secrets in str, repr, and get_masked_dict."""
    emb = EmbedderConfig(api_key="hf_live_token_abc")
    pg = PGVectorConfig(password="postgres_live_pw_xyz")
    app = AppConfig(embedder=emb, pgvector=pg)

    # Representations must NOT leak either secret
    repr_str = repr(app)
    str_str = str(app)

    assert "hf_live_token_abc" not in repr_str
    assert "postgres_live_pw_xyz" not in repr_str
    assert "hf_live_token_abc" not in str_str
    assert "postgres_live_pw_xyz" not in str_str

    # Masked dict verification
    masked_dict = app.get_masked_dict()
    assert masked_dict["embedder"]["api_key"] == "hf_l...******"
    assert masked_dict["pgvector"]["password"] == "post...******"
