import os
import sys
import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# Ensure repo root is in sys.path for direct execution or script invocation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from .routes import router
except ImportError:
    from src.api.routes import router

from src.telemetry.metrics import metrics_registry
from src.telemetry.tracing import tracer, trace_span, generate_trace_id, generate_span_id
from src.telemetry.logging_config import setup_telemetry_logging

# Initialize telemetry logging
setup_telemetry_logging(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="Kitchome RAG Intelligence Platform API",
    description=(
        "Production-grade Multi-Tenant RAG Service with Local Onboarding, "
        "Strict Single-Document Ingestion, Clearance Level ABAC, and Role Governance."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local Streamlit and frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    """
    Middleware that correlates traces, injects W3C and X-Trace-Id headers,
    and records Prometheus HTTP request metrics.
    """
    start_time = time.time()
    trace_id = request.headers.get("X-Trace-Id")
    if not trace_id:
        traceparent = request.headers.get("traceparent")
        if traceparent and len(traceparent.split("-")) >= 4:
            trace_id = traceparent.split("-")[1]
        else:
            trace_id = generate_trace_id()

    endpoint = request.url.path
    method = request.method

    with trace_span(f"http_{method.lower()}", {"http.method": method, "http.path": endpoint}, trace_id=trace_id) as span:
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            status_code = str(response.status_code)
            
            # Record Prometheus metrics
            metrics_registry.http_requests_total.inc(method=method, endpoint=endpoint, status_code=status_code)
            metrics_registry.http_request_duration_seconds.observe(duration, method=method, endpoint=endpoint)
            
            # Inject tracing headers
            response.headers["X-Trace-Id"] = trace_id
            response.headers["traceparent"] = f"00-{trace_id}-{span.span_id}-01"
            return response
        except Exception as exc:
            duration = time.time() - start_time
            metrics_registry.http_requests_total.inc(method=method, endpoint=endpoint, status_code="500")
            metrics_registry.http_request_duration_seconds.observe(duration, method=method, endpoint=endpoint)
            span.set_error(str(exc))
            raise exc

app.include_router(router)

@app.get("/metrics", tags=["Observability"])
def prometheus_metrics():
    """Exposes application metrics in standard Prometheus/OpenMetrics text format."""
    content = metrics_registry.generate_metrics_text()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")

@app.get("/healthz", tags=["System"])
def health_check():
    """Service liveness probe."""
    return {"status": "ok", "service": "kitchome-rag", "version": "1.0.0"}

@app.get("/", tags=["System"])
def root():
    """Root metadata endpoint."""
    return {
        "service": "Kitchome RAG Intelligence API",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics",
        "endpoints": {
            "query": "POST /api/v1/query",
            "ingest": "POST /api/v1/ingest",
            "ingest_file": "POST /api/v1/ingest/file",
            "ingest_history": "GET /api/v1/ingest/history",
            "ingest_retry": "POST /api/v1/ingest/retry/{job_id}",
            "profile": "GET /api/v1/auth/me",
            "admin_pending": "GET /api/v1/admin/pending",
            "admin_approve": "POST /api/v1/admin/approve",
            "admin_reject": "POST /api/v1/admin/reject",
            "admin_update_access": "POST /api/v1/admin/update-access",
            "admin_users": "GET /api/v1/admin/users"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


