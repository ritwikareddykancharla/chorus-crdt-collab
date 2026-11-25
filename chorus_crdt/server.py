"""Local HTTP Server & AWS CloudFront `/api/*` Lambda Function URL Handler for ChorusCRDT."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from .ai_collaborator import ChorusWorkspace
from .fuzzer import simulate_partition_scenario

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"

WORKSPACE = ChorusWorkspace()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Pure request router shared by local HTTP server and AWS Lambda `/api/*` behind CloudFront."""
    try:
        if method == "GET" and path == "/api/health":
            storage = WORKSPACE.store.get_storage_stats()
            return 200, {
                "status": "ok",
                "service": "chorus-crdt-collab",
                "delivery": os.environ.get("CHORUS_DELIVERY", "cloudfront-s3-oac + lambda-api-origin"),
                "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
                "bedrock_model": os.environ.get("CHORUS_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
                "python_version": platform.python_version(),
                "request_id": request_id,
                "documents_loaded": len(WORKSPACE.docs),
                "sqlite_wal_frames": storage["wal_frames"],
                "sqlite_checkpoints": storage["checkpoints_count"],
            }

        if method == "GET" and path == "/api/doc/state":
            doc_id = str(query.get("doc_id") or "doc-rfc-sync").strip()
            snap = WORKSPACE.get_workspace_state(doc_id=doc_id)
            snap["request_id"] = request_id
            return 200, snap

        if method == "POST" and path == "/api/doc/coedit":
            doc_id = str(body.get("doc_id") or "doc-rfc-sync").strip()
            section_key = body.get("section_key")
            ai_mode = str(body.get("ai_mode") or "tighten_metrics").strip()
            custom_instruction = str(body.get("custom_instruction") or "").strip()
            concurrent_human_text = str(
                body.get(
                    "concurrent_human_text",
                    "[Maya live note: Verified 0 packet loss across 4 Availability Zones.]\n",
                )
            )
            human_position = str(body.get("human_position") or "above_anchor").strip()
            out = WORKSPACE.coedit_document(
                doc_id=doc_id,
                section_key=str(section_key).strip() if section_key else None,
                ai_mode=ai_mode,
                custom_instruction=custom_instruction,
                concurrent_human_text=concurrent_human_text,
                human_position=human_position,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/sync/delta":
            doc_id = str(body.get("doc_id") or "doc-rfc-sync").strip()
            incremental_edit = str(
                body.get(
                    "incremental_edit",
                    "\n## 4. Operational Readiness\nAlarms trigger if p99 CRDT fan-out exceeds 45 ms.",
                )
            )
            run_compaction = bool(body.get("run_sqlite_compaction", True))
            out = WORKSPACE.inspect_binary_delta(
                doc_id=doc_id,
                incremental_edit=incremental_edit,
                run_sqlite_compaction=run_compaction,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/undo/actor":
            doc_id = str(body.get("doc_id") or "doc-undo-spec").strip()
            action = str(body.get("action") or "undo").strip()
            if action == "stage":
                out = WORKSPACE.stage_interleaved_edits(doc_id=doc_id)
                out["request_id"] = request_id
                return 200, out

            actor_id = str(body.get("actor_id") or "bedrock-nova").strip()
            tx_id = body.get("tx_id")
            out = WORKSPACE.selective_undo(
                doc_id=doc_id,
                actor_id=actor_id,
                tx_id=str(tx_id).strip() if tx_id else None,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/fuzz/run":
            seed = int(body.get("seed") or 10599520)
            scenario_mode = str(body.get("scenario_mode") or "split_brain_rfc").strip()
            num_fuzz_seeds = int(body.get("num_fuzz_seeds") or 60)
            out = simulate_partition_scenario(
                seed=seed,
                scenario_mode=scenario_mode,
                num_fuzz_seeds=num_fuzz_seeds,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "GET" and path == "/api/benchmarks":
            bench = WORKSPACE.load_benchmarks()
            bench["request_id"] = request_id
            return 200, bench

        if method == "POST" and path == "/api/reset":
            WORKSPACE.reset()
            state = WORKSPACE.get_workspace_state("doc-rfc-sync")
            return 200, {
                "status": "reset_complete",
                "service": "chorus-crdt-collab",
                "request_id": request_id,
                "workspace": state,
            }

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except KeyError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None}
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalChorusHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 3099) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalChorusHandler)
    print(f"ChorusCRDT Multiplayer Studio running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "3099"))
    run_local_server(port=port_num)
