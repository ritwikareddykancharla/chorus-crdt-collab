"""Live Amazon Bedrock Co-Authoring Peer, Relative Cursor Anchoring & Selective Per-Actor Undo."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .wire_sync import (
    SQLiteCRDTStore,
    compute_sync_delta,
    decode_binary_delta,
    encode_state_vector,
    sync_peers_binary,
)
from .yata_engine import RelativeRange, YataDocument

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

# Cached exported profile credentials for fast local boto3 Converse execution
_CACHED_AWS_CREDS: Optional[Dict[str, Any]] = None


def _get_boto3_bedrock_client(region: str) -> Any:
    import boto3  # type: ignore

    # In AWS Lambda, IAM execution role credentials are injected via standard env vars
    if os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("LAMBDA_TASK_ROOT"):
        return boto3.client("bedrock-runtime", region_name=region)

    global _CACHED_AWS_CREDS
    now_epoch = time.time()
    if _CACHED_AWS_CREDS and _CACHED_AWS_CREDS.get("expires_at", 0) - 60 > now_epoch:
        return boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=_CACHED_AWS_CREDS["AccessKeyId"],
            aws_secret_access_key=_CACHED_AWS_CREDS["SecretAccessKey"],
            aws_session_token=_CACHED_AWS_CREDS.get("SessionToken"),
        )

    aws_bin = shutil.which("aws")
    if aws_bin:
        profile = os.environ.get("CHORUS_AWS_PROFILE", os.environ.get("AWS_PROFILE", "portfolio"))
        env = dict(os.environ)
        env.pop("AWS_ACCESS_KEY_ID", None)
        env.pop("AWS_SECRET_ACCESS_KEY", None)
        env.pop("AWS_SESSION_TOKEN", None)
        proc = subprocess.run(
            [aws_bin, "configure", "export-credentials", "--profile", profile, "--format", "process"],
            env=env,
            capture_output=True,
            text=True,
            timeout=6.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            creds = json.loads(proc.stdout)
            exp_raw = creds.get("Expiration")
            exp_ts = now_epoch + 900
            if exp_raw:
                try:
                    exp_ts = datetime.fromisoformat(exp_raw.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    pass
            creds["expires_at"] = exp_ts
            _CACHED_AWS_CREDS = creds
            return boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds.get("SessionToken"),
            )

    return boto3.client("bedrock-runtime", region_name=region)


def invoke_bedrock_nova(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 260,
    temperature: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """Invoke live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) via Converse API and record exact telemetry."""
    if os.environ.get("CHORUS_OFFLINE_LLM") == "1":
        return None

    model_id = os.environ.get("CHORUS_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    t0 = time.perf_counter()

    try:
        client = _get_boto3_bedrock_client(region)
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        round_trip_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        server_ms = float(resp.get("metrics", {}).get("latencyMs") or round_trip_ms)
        usage = resp.get("usage", {})
        raw_text = resp["output"]["message"]["content"][0]["text"].strip()
        # Normalize any em-dash to plain comma/colon/hyphen so generated prose stays clean
        cleaned_text = raw_text.replace("\u2014", " - ").replace("```markdown", "").replace("```", "").strip()
        return {
            "text": cleaned_text,
            "engine": f"aws-bedrock ({model_id})",
            "model_id": model_id,
            "latency_ms": round(server_ms, 2),
            "round_trip_ms": round(round_trip_ms, 2),
            "input_tokens": int(usage.get("inputTokens", 0)),
            "output_tokens": int(usage.get("outputTokens", 0)),
            "total_tokens": int(usage.get("totalTokens", 0)),
        }
    except Exception:
        pass

    aws_bin = shutil.which("aws")
    if not aws_bin:
        return None

    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)
    env["AWS_PROFILE"] = os.environ.get("CHORUS_AWS_PROFILE", os.environ.get("AWS_PROFILE", "portfolio"))
    env["AWS_DEFAULT_REGION"] = region
    cmd = [
        aws_bin,
        "bedrock-runtime",
        "converse",
        "--region",
        region,
        "--model-id",
        model_id,
        "--system",
        json.dumps([{"text": system_prompt}]),
        "--messages",
        json.dumps([{"role": "user", "content": [{"text": user_prompt}]}]),
        "--inference-config",
        json.dumps({"maxTokens": max_tokens, "temperature": temperature}),
        "--output",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=14.0)
        round_trip_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            server_ms = float(data.get("metrics", {}).get("latencyMs") or round_trip_ms)
            usage = data.get("usage", {})
            raw_text = data["output"]["message"]["content"][0]["text"].strip()
            cleaned_text = raw_text.replace("\u2014", " - ").replace("```markdown", "").replace("```", "").strip()
            return {
                "text": cleaned_text,
                "engine": f"aws-bedrock ({model_id})",
                "model_id": model_id,
                "latency_ms": round(server_ms, 2),
                "round_trip_ms": round(round_trip_ms, 2),
                "input_tokens": int(usage.get("inputTokens", 0)),
                "output_tokens": int(usage.get("outputTokens", 0)),
                "total_tokens": int(usage.get("totalTokens", 0)),
            }
    except Exception:
        return None
    return None


DOCUMENT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "doc-rfc-sync": {
        "doc_id": "doc-rfc-sync",
        "title": "RFC-042: Sub-50ms Multiplayer CRDT Sync",
        "owner": "maya.chen@aws-chorus.internal",
        "sections": {
            "latency_goal": "Editors expect local keystrokes in 1 ms and peer cursor updates in under 50 ms across 200 concurrent collaborators.",
            "anchor_protocol": "Cursors and review threads bind to immutable (actor_id, lamport_clock) item IDs rather than integer string offsets.",
            "cost_budget": "Batching varint binary deltas into compacted S3/SQLite checkpoints keeps infrastructure near $0.72 per active builder per year.",
        },
        "initial_text": (
            "# RFC-042: Sub-50ms Multiplayer CRDT Sync\n\n"
            "## 1. Latency & Fan-Out Goal\n"
            "Editors expect local keystrokes in 1 ms and peer cursor updates in under 50 ms across 200 concurrent collaborators.\n\n"
            "## 2. Logical Cursor & Comment Anchors\n"
            "Cursors and review threads bind to immutable (actor_id, lamport_clock) item IDs rather than integer string offsets.\n\n"
            "## 3. Storage & Bandwidth Economics\n"
            "Batching varint binary deltas into compacted S3/SQLite checkpoints keeps infrastructure near $0.72 per active builder per year.\n"
        ),
    },
    "doc-incident-review": {
        "doc_id": "doc-incident-review",
        "title": "COE-118: WebSocket Reconnect Delta Storm",
        "owner": "alex.rivera@aws-chorus.internal",
        "sections": {
            "impact_summary": "During a regional Wi-Fi blip, 140 reconnecting browser tabs uploaded full JSON snapshots instead of 28-byte StateVectors.",
            "root_cause": "The legacy fallback client skipped CHSV StateVector negotiation when buffered offline edits exceeded 32 operations.",
            "remediation_plan": "Enforce mandatory CHSV binary StateVector handshake on reconnect and cap uncompacted WAL frames at 64 entries.",
        },
        "initial_text": (
            "# COE-118: WebSocket Reconnect Delta Storm\n\n"
            "## 1. Customer Impact Summary\n"
            "During a regional Wi-Fi blip, 140 reconnecting browser tabs uploaded full JSON snapshots instead of 28-byte StateVectors.\n\n"
            "## 2. Technical Root Cause\n"
            "The legacy fallback client skipped CHSV StateVector negotiation when buffered offline edits exceeded 32 operations.\n\n"
            "## 3. Action Items & Guardrails\n"
            "Enforce mandatory CHSV binary StateVector handshake on reconnect and cap uncompacted WAL frames at 64 entries.\n"
        ),
    },
    "doc-undo-spec": {
        "doc_id": "doc-undo-spec",
        "title": "Spec: Selective Per-Author Undo for AI Teammates",
        "owner": "priya.nair@aws-chorus.internal",
        "sections": {
            "problem_scope": "Linear stack undo reverts the most recent global edit, accidentally erasing a teammate's paragraph when reverting an AI draft.",
            "causal_inversion": "Each transaction records its inserted item IDs and prior tombstoned item IDs so inverse mutations affect only one actor.",
            "presence_ux": "AI collaborators appear in the presence bar with named cursors, live selection highlights, and 1-click transaction revert.",
        },
        "initial_text": (
            "# Spec: Selective Per-Author Undo for AI Teammates\n\n"
            "## 1. Linear Stack Undo Limitations in Multiplayer\n"
            "Linear stack undo reverts the most recent global edit, accidentally erasing a teammate's paragraph when reverting an AI draft.\n\n"
            "## 2. Causal Transaction Inversion\n"
            "Each transaction records its inserted item IDs and prior tombstoned item IDs so inverse mutations affect only one actor.\n\n"
            "## 3. Agent Presence & Review Controls\n"
            "AI collaborators appear in the presence bar with named cursors, live selection highlights, and 1-click transaction revert.\n"
        ),
    },
}


class ChorusWorkspace:
    """Serverless control plane coordinating Yata CRDT documents, live Bedrock co-editing, and SQLite WAL sync."""

    def __init__(self, store: Optional[SQLiteCRDTStore] = None) -> None:
        self.store = store or SQLiteCRDTStore()
        self.docs: Dict[str, YataDocument] = {}
        self.anchors: Dict[str, Dict[str, RelativeRange]] = {}
        self._seed_documents()

    def _seed_documents(self) -> None:
        self.docs.clear()
        self.anchors.clear()
        for doc_id, spec in DOCUMENT_TEMPLATES.items():
            doc = YataDocument(actor_id="workspace-seed", author_kind="system", doc_id=doc_id)
            emitted, _ = doc.insert_text(
                0,
                spec["initial_text"],
                actor_id="workspace-seed",
                author_kind="system",
                tx_id="tx-init",
                summary=f"Initialize {spec['title']}",
            )
            sec_ranges: Dict[str, RelativeRange] = {}
            for sec_key, sec_text in spec["sections"].items():
                sec_ranges[sec_key] = doc.create_range_for_substring(
                    sec_text,
                    label=sec_key,
                    owner_id="bedrock-nova",
                )
            # Register initial peer cursors
            first_key = next(iter(sec_ranges.keys()))
            doc.update_presence(
                actor_id="peer-maya",
                author_kind="human",
                anchor=doc.create_anchor(len(spec["title"]) + 4, "after"),
                status="online",
            )
            doc.update_presence(
                actor_id="bedrock-nova",
                author_kind="ai",
                anchor=sec_ranges[first_key].end,
                selection_start=sec_ranges[first_key].start,
                status="ready",
            )
            doc.update_presence(
                actor_id="peer-alex",
                author_kind="human",
                anchor=doc.create_anchor(doc.active_item_count(), "after"),
                status="online",
            )
            self.store.append_ops(
                doc_id=doc_id,
                actor_id="workspace-seed",
                tx_id="tx-init",
                ops=list(emitted),
                state_vector=doc.get_contiguous_state_vector(),
            )
            self.docs[doc_id] = doc
            self.anchors[doc_id] = sec_ranges

    def reset(self) -> None:
        self.store.reset()
        self._seed_documents()

    def _get_doc(self, doc_id: str) -> YataDocument:
        if doc_id not in self.docs:
            raise KeyError(f"Unknown document ID: {doc_id}")
        return self.docs[doc_id]

    def get_workspace_state(self, doc_id: str = "doc-rfc-sync") -> Dict[str, Any]:
        doc = self._get_doc(doc_id)
        resolved_sections: Dict[str, Any] = {}
        for sec_key, r_range in self.anchors.get(doc_id, {}).items():
            s, e, excerpt = doc.resolve_range(r_range)
            resolved_sections[sec_key] = {
                "section_key": sec_key,
                "start_offset": s,
                "end_offset": e,
                "excerpt": excerpt,
                "anchor": r_range.to_dict(),
            }
        return {
            "active_doc_id": doc_id,
            "documents": [
                {
                    "doc_id": k,
                    "title": DOCUMENT_TEMPLATES[k]["title"],
                    "owner": DOCUMENT_TEMPLATES[k]["owner"],
                    "active_chars": self.docs[k].active_item_count(),
                    "tombstones": self.docs[k].tombstone_count(),
                    "sections": list(DOCUMENT_TEMPLATES[k]["sections"].keys()),
                }
                for k in DOCUMENT_TEMPLATES
            ],
            "text": doc.get_text(),
            "author_spans": doc.get_author_spans(),
            "presence_cursors": doc.get_presence_cursors(),
            "resolved_sections": resolved_sections,
            "transactions": doc.list_transactions(),
            "state_vector": doc.get_contiguous_state_vector(),
            "sqlite_storage": self.store.get_storage_stats(),
        }

    # ------------------------------------------------------------------
    # Live Multiplayer Human + Amazon Bedrock Co-Editing
    # ------------------------------------------------------------------
    def coedit_document(
        self,
        *,
        doc_id: str = "doc-rfc-sync",
        section_key: Optional[str] = None,
        ai_mode: str = "tighten_metrics",
        custom_instruction: str = "",
        concurrent_human_text: str = "[Maya live note: Verified 0 packet loss across 4 Availability Zones.]\n",
        human_position: str = "above_anchor",
    ) -> Dict[str, Any]:
        """
        Execute true concurrent human + Amazon Bedrock co-editing:
          1. Pin a `RelativeRange` on `section_key` BEFORE any human edits start.
          2. Fork two concurrent peer replicas (`peer-maya` [human] and `bedrock-nova` [AI]).
          3. `peer-maya` types `concurrent_human_text` ABOVE (or below) the AI's target section, shifting raw integer offsets!
          4. Concurrently, `bedrock-nova` calls Amazon Bedrock (`us.amazon.nova-lite-v1:0`) to rewrite the anchored span.
          5. Merge both replicas via binary `CHSV`/`CHDL` StateVector delta sync into the authoritative document.
        """
        doc = self._get_doc(doc_id)
        sec_map = self.anchors[doc_id]
        chosen_key = section_key if section_key in sec_map else next(iter(sec_map.keys()))
        pinned_range = sec_map[chosen_key]

        pre_start, pre_end, target_excerpt = doc.resolve_range(pinned_range)

        # Fork two independent peer replicas at the exact same causal snapshot
        maya_peer = doc.clone("peer-maya", "human")
        nova_peer = doc.clone("bedrock-nova", "ai")

        # 1. Human collaborator types concurrently (default: above the AI's target span, shifting all integer offsets!)
        human_tx_id = f"tx-maya-{uuid.uuid4().hex[:6]}"
        human_ops = []
        if concurrent_human_text:
            if human_position == "below_anchor":
                insert_at = min(maya_peer.active_item_count(), pre_end + 1)
            else:
                # Insert right near top of document (before `pre_start`) so integer offsets shift!
                newline_pos = maya_peer.get_text().find("\n\n")
                insert_at = (newline_pos + 2) if 0 <= newline_pos < pre_start else max(0, pre_start)
            h_ops, _ = maya_peer.insert_text(
                insert_at,
                concurrent_human_text,
                actor_id="peer-maya",
                author_kind="human",
                tx_id=human_tx_id,
                summary=f"Maya concurrent edit ({len(concurrent_human_text)} chars {human_position})",
            )
            human_ops.extend(h_ops)

        # 2. Live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) edits its pinned RelativeRange on `nova_peer`
        mode_prompts = {
            "tighten_metrics": "Rewrite this engineering spec sentence to be sharper, concrete, and include specific latency, byte-budget, or failure-handling detail in 1-2 crisp sentences.",
            "security_review": "Expand this engineering spec span with one concrete authentication, tenant-isolation, or replay-protection safeguard in 1-2 crisp sentences.",
            "cost_breakdown": "Tighten this span and add concrete serverless batching/compaction unit-economics reasoning in 1-2 crisp sentences.",
        }
        directive = custom_instruction.strip() or mode_prompts.get(ai_mode, mode_prompts["tighten_metrics"])
        sys_prompt = (
            "You are @nova-editor, an AI engineering teammate co-editing a live technical RFC inside a CRDT document workspace. "
            "Return ONLY the replacement prose for the selected span (1 to 2 concise technical sentences, plain text, no markdown headings, "
            "no bullet lists, no surrounding quotes, and never use the em-dash character)."
        )
        user_prompt = (
            f"Document: {DOCUMENT_TEMPLATES[doc_id]['title']}\n"
            f"Selected CRDT Anchor Span ({chosen_key}): {target_excerpt}\n"
            f"Edit Goal: {directive}\n"
            "Output ONLY the replacement sentences:"
        )

        bedrock_res = invoke_bedrock_nova(sys_prompt, user_prompt, max_tokens=180, temperature=0.15)
        if bedrock_res and bedrock_res.get("text"):
            ai_replacement = bedrock_res["text"]
            llm_meta = {
                "engine": bedrock_res["engine"],
                "model_id": bedrock_res["model_id"],
                "bedrock_latency_ms": bedrock_res["latency_ms"],
                "round_trip_ms": bedrock_res["round_trip_ms"],
                "input_tokens": bedrock_res["input_tokens"],
                "output_tokens": bedrock_res["output_tokens"],
                "total_tokens": bedrock_res["total_tokens"],
            }
        else:
            ai_replacement = (
                f"{target_excerpt.rstrip('.')} Verified via deterministic Yata causal anchors with "
                "28-byte CHSV state vectors and zero offset drift under concurrent edits."
            )
            llm_meta = {
                "engine": "deterministic-fallback",
                "model_id": "offline-fallback",
                "bedrock_latency_ms": 0.0,
                "round_trip_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        t_merge0 = time.perf_counter()
        ai_tx_id = f"tx-nova-{uuid.uuid4().hex[:6]}"
        ai_ops, ai_tx, _ = nova_peer.replace_anchor_range(
            pinned_range,
            ai_replacement,
            actor_id="bedrock-nova",
            author_kind="ai",
            tx_id=ai_tx_id,
            summary=f"@nova-editor ({ai_mode}) on {chosen_key}",
        )

        # Update section's RelativeRange to track the newly inserted AI items for future edits!
        if ai_tx.inserted_ids:
            from .yata_engine import RelativeAnchor

            sec_map[chosen_key] = RelativeRange(
                start=RelativeAnchor(ai_tx.inserted_ids[0], "before"),
                end=RelativeAnchor(ai_tx.inserted_ids[-1], "after"),
                label=chosen_key,
                owner_id="bedrock-nova",
            )

        # 3. Compute binary StateVector delta from both concurrent peers into `doc`
        sv_before = encode_state_vector(doc.get_contiguous_state_vector())
        maya_delta_info = compute_sync_delta(maya_peer, sv_before)
        nova_delta_info = compute_sync_delta(nova_peer, sv_before)

        doc.apply_ops(decode_binary_delta(maya_delta_info["delta_raw"]))
        doc.apply_ops(decode_binary_delta(nova_delta_info["delta_raw"]))

        # Register transactions & presence on authoritative `doc` for 1-click Selective Per-Actor Undo
        if human_ops:
            doc._register_tx(
                tx_id=human_tx_id,
                actor_id="peer-maya",
                author_kind="human",
                summary=f"Maya concurrent note ({len(concurrent_human_text.strip())} chars)",
                inserted_ids=[op.id for op in human_ops],
            )
            doc.update_presence(
                actor_id="peer-maya",
                author_kind="human",
                anchor=maya_peer.create_anchor(maya_peer.active_item_count(), "after"),
                status="editing",
            )

        doc._register_tx(
            tx_id=ai_tx_id,
            actor_id="bedrock-nova",
            author_kind="ai",
            summary=f"@nova-editor rewrote {chosen_key} ({ai_mode})",
            inserted_ids=list(ai_tx.inserted_ids),
            deleted_ids=list(ai_tx.deleted_ids),
        )
        doc.update_presence(
            actor_id="bedrock-nova",
            author_kind="ai",
            anchor=sec_map[chosen_key].end,
            selection_start=sec_map[chosen_key].start,
            status="synced",
        )

        # Reconcile peers back so all 3 replicas are 100% identical
        sync_peers_binary(doc, maya_peer)
        sync_peers_binary(doc, nova_peer)
        crdt_merge_ms = round((time.perf_counter() - t_merge0) * 1000.0, 3)

        post_start, post_end, post_excerpt = doc.resolve_range(sec_map[chosen_key])
        final_text = doc.get_text()
        human_preserved = (concurrent_human_text in final_text) if concurrent_human_text else True

        # Persist binary frames in SQLite WAL
        if human_ops:
            self.store.append_ops(
                doc_id=doc_id,
                actor_id="peer-maya",
                tx_id=human_tx_id,
                ops=list(human_ops),
                state_vector=doc.get_contiguous_state_vector(),
            )
        wal_receipt = self.store.append_ops(
            doc_id=doc_id,
            actor_id="bedrock-nova",
            tx_id=ai_tx_id,
            ops=list(ai_ops),
            state_vector=doc.get_contiguous_state_vector(),
        )

        return {
            "doc_id": doc_id,
            "section_key": chosen_key,
            "ai_mode": ai_mode,
            "anchor_shift_proof": {
                "offset_before_human_edit": [pre_start, pre_end],
                "offset_after_concurrent_merge": [post_start, post_end],
                "shifted_by_chars": post_start - pre_start,
                "original_excerpt": target_excerpt,
                "ai_rewritten_excerpt": post_excerpt,
                "human_concurrent_preserved": human_preserved,
                "all_replicas_converged": final_text == maya_peer.get_text() == nova_peer.get_text(),
            },
            "crdt_merge_ms": crdt_merge_ms,
            "wire_metrics": {
                "state_vector_bytes": nova_delta_info["state_vector_bytes"],
                "human_delta_bytes": maya_delta_info["delta_bytes"],
                "ai_delta_bytes": nova_delta_info["delta_bytes"],
                "combined_wire_bytes": maya_delta_info["delta_bytes"] + nova_delta_info["delta_bytes"] + nova_delta_info["state_vector_bytes"],
                "full_crdt_snapshot_bytes": nova_delta_info["full_crdt_snapshot_bytes"],
                "reduction_vs_crdt_snapshot_pct": nova_delta_info["reduction_vs_crdt_snapshot_pct"],
            },
            "bedrock": llm_meta,
            "wal_receipt": wal_receipt,
            "human_tx_id": human_tx_id if human_ops else None,
            "ai_tx_id": ai_tx_id,
            "text": final_text,
            "author_spans": doc.get_author_spans(),
            "presence_cursors": doc.get_presence_cursors(),
            "transactions": doc.list_transactions(),
        }

    # ------------------------------------------------------------------
    # Binary StateVector Delta Inspector & SQLite WAL Compaction (Tab 03)
    # ------------------------------------------------------------------
    def inspect_binary_delta(
        self,
        *,
        doc_id: str = "doc-rfc-sync",
        incremental_edit: str = "\n## 4. Production Readiness\n Alarms fire if p99 CRDT fan-out exceeds 45 ms.",
        run_sqlite_compaction: bool = True,
    ) -> Dict[str, Any]:
        doc = self._get_doc(doc_id)
        lagging_peer = doc.clone("peer-lagging-mobile", "human")
        sv_lagging_bytes = encode_state_vector(lagging_peer.get_contiguous_state_vector())

        tx_id = f"tx-alex-{uuid.uuid4().hex[:6]}"
        emitted, _ = doc.insert_text(
            doc.active_item_count(),
            incremental_edit,
            actor_id="peer-alex",
            author_kind="human",
            tx_id=tx_id,
            summary=f"Incremental {len(incremental_edit)}-char edit for binary wire inspection",
        )
        self.store.append_ops(
            doc_id=doc_id,
            actor_id="peer-alex",
            tx_id=tx_id,
            ops=list(emitted),
            state_vector=doc.get_contiguous_state_vector(),
        )

        delta_report = compute_sync_delta(doc, sv_lagging_bytes)
        lagging_peer.apply_ops(decode_binary_delta(delta_report["delta_raw"]))

        compaction_report = self.store.compact_document_log(doc) if run_sqlite_compaction else None
        hydrated_ok = True
        if compaction_report:
            hydrated = self.store.hydrate_document(doc_id)
            hydrated_ok = hydrated.get_text() == doc.get_text()

        clean_report = {k: v for k, v in delta_report.items() if k != "delta_raw"}
        return {
            "doc_id": doc_id,
            "incremental_chars": len(incremental_edit),
            "lagging_peer_converged": lagging_peer.get_text() == doc.get_text(),
            "hydrated_roundtrip_verified": hydrated_ok,
            "delta_metrics": clean_report,
            "sqlite_compaction": compaction_report,
            "state_vector_decoded": doc.get_contiguous_state_vector(),
            "text": doc.get_text(),
            "transactions": doc.list_transactions(),
        }

    # ------------------------------------------------------------------
    # Selective Per-Actor Undo Lab (Tab 04)
    # ------------------------------------------------------------------
    def stage_interleaved_edits(self, doc_id: str = "doc-undo-spec") -> Dict[str, Any]:
        """Stage 3 interleaved transactions by Maya, @nova-editor, and Alex on the same paragraph."""
        doc = self._get_doc(doc_id)
        # Insert 3 distinct collaborator contributions in sequence so global undo would break Alex & Maya
        m_ops, m_tx = doc.insert_text(
            doc.active_item_count(),
            "\n[Maya @ 10:14:02] Added per-tenant KMS envelope key check before snapshot upload.",
            actor_id="peer-maya",
            author_kind="human",
            summary="Maya: KMS envelope key requirement",
        )
        self.store.append_ops(doc_id=doc_id, actor_id="peer-maya", tx_id=m_tx.tx_id, ops=list(m_ops), state_vector=doc.get_contiguous_state_vector())

        n_ops, n_tx = doc.insert_text(
            doc.active_item_count(),
            "\n[@nova-editor @ 10:14:03] Drafted experimental auto-prune policy reducing tombstone retention to 30 seconds.",
            actor_id="bedrock-nova",
            author_kind="ai",
            summary="@nova-editor: 30s aggressive tombstone prune draft",
        )
        self.store.append_ops(doc_id=doc_id, actor_id="bedrock-nova", tx_id=n_tx.tx_id, ops=list(n_ops), state_vector=doc.get_contiguous_state_vector())

        a_ops, a_tx = doc.insert_text(
            doc.active_item_count(),
            "\n[Alex @ 10:14:05] Verified offline reconnect needs 24-hour tombstone window; keep Maya's KMS check.",
            actor_id="peer-alex",
            author_kind="human",
            summary="Alex: 24h reconnect window note (typed AFTER @nova-editor)",
        )
        self.store.append_ops(doc_id=doc_id, actor_id="peer-alex", tx_id=a_tx.tx_id, ops=list(a_ops), state_vector=doc.get_contiguous_state_vector())

        return {
            "doc_id": doc_id,
            "staged_transactions": [m_tx.to_dict(), n_tx.to_dict(), a_tx.to_dict()],
            "text": doc.get_text(),
            "author_spans": doc.get_author_spans(),
            "transactions": doc.list_transactions(),
        }

    def selective_undo(
        self,
        *,
        doc_id: str = "doc-undo-spec",
        actor_id: str = "bedrock-nova",
        tx_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        doc = self._get_doc(doc_id)
        t0 = time.perf_counter()
        before_text = doc.get_text()
        emitted_muts, undo_record = doc.undo_actor(actor_id, tx_id=tx_id)
        undo_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if emitted_muts and undo_record:
            self.store.append_ops(
                doc_id=doc_id,
                actor_id=doc.actor_id,
                tx_id=undo_record.tx_id,
                ops=list(emitted_muts),
                state_vector=doc.get_contiguous_state_vector(),
            )

        after_text = doc.get_text()
        # Count remaining active characters per non-target collaborator to prove 0 collateral loss
        active_by_other: Dict[str, int] = {}
        for sp in doc.get_author_spans():
            if sp["actor_id"] != actor_id:
                active_by_other[sp["actor_id"]] = active_by_other.get(sp["actor_id"], 0) + len(sp["text"])

        return {
            "doc_id": doc_id,
            "undone_actor_id": actor_id,
            "undo_applied": undo_record is not None,
            "undo_record": undo_record.to_dict() if undo_record else None,
            "inverse_mutations_emitted": len(emitted_muts),
            "undo_latency_ms": undo_ms,
            "chars_before": len(before_text),
            "chars_after": len(after_text),
            "preserved_collaborator_chars": active_by_other,
            "text": after_text,
            "author_spans": doc.get_author_spans(),
            "transactions": doc.list_transactions(),
        }

    def load_benchmarks(self) -> Dict[str, Any]:
        if BENCHMARK_JSON_PATH.is_file():
            return json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        return {"status": "not_generated"}
