"""Comprehensive Unit Tests for ChorusCRDT (Yata CRDT, Relative Anchors, Binary Wire Sync, Fuzzer, Selective Undo & API)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from chorus_crdt.ai_collaborator import ChorusWorkspace
from chorus_crdt.fuzzer import run_convergence_fuzz_suite, simulate_partition_scenario
from chorus_crdt.server import lambda_handler, route_api_request
from chorus_crdt.wire_sync import (
    SQLiteCRDTStore,
    compute_sync_delta,
    decode_binary_delta,
    decode_state_vector,
    decode_varint,
    encode_binary_delta,
    encode_state_vector,
    encode_varint,
    sync_peers_binary,
)
from chorus_crdt.yata_engine import YataDocument


class TestChorusCRDT(unittest.TestCase):
    def test_01_yata_concurrent_same_offset_non_interleaving(self) -> None:
        """Concurrent multi-char word insertions at the exact same offset converge with zero scrambled letters."""
        seed = YataDocument(actor_id="seed", author_kind="system")
        seed.insert_text(0, "Hello !")

        peer_a = seed.clone("peer-alice", "human")
        peer_b = seed.clone("peer-bob", "human")
        peer_c = seed.clone("bedrock-nova", "ai")

        ops_a, _ = peer_a.insert_text(6, "collaborative ")
        ops_b, _ = peer_b.insert_text(6, "distributed ")
        ops_c, _ = peer_c.insert_text(6, "intelligent ")

        # Apply in 3 completely different orders across the 3 replicas
        peer_a.apply_ops(ops_b + ops_c)
        peer_b.apply_ops(ops_c + ops_a)
        peer_c.apply_ops(ops_a + ops_b)

        self.assertEqual(peer_a.get_text(), peer_b.get_text())
        self.assertEqual(peer_b.get_text(), peer_c.get_text())
        merged = peer_a.get_text()
        self.assertIn("collaborative ", merged)
        self.assertIn("distributed ", merged)
        self.assertIn("intelligent ", merged)

    def test_02_out_of_order_causal_buffering_and_duplicate_idempotency(self) -> None:
        """Reversed packet arrival order buffers safely until causal origins arrive; duplicates are no-ops."""
        doc_a = YataDocument(actor_id="peer-a")
        ops1, _ = doc_a.insert_text(0, "Chorus CRDT")
        ops2, _ = doc_a.delete_range(7, 11)
        ops3, _ = doc_a.insert_text(7, "Workspace")

        all_ops = list(ops1) + list(ops2) + list(ops3)
        reversed_ops = list(reversed(all_ops))

        doc_b = YataDocument(actor_id="peer-b")
        # Deliver completely reversed + replay duplicates
        doc_b.apply_ops(reversed_ops)
        doc_b.apply_ops(reversed_ops[:8])

        self.assertEqual(doc_b.pending_count(), 0)
        self.assertEqual(doc_b.get_text(), "Chorus Workspace")
        self.assertEqual(doc_a.get_text(), doc_b.get_text())

    def test_03_relative_anchors_survive_upstream_inserts_and_deletes(self) -> None:
        """RelativeRange stays pinned to exact target words when paragraphs are added or deleted above it."""
        doc = YataDocument(actor_id="peer-maya")
        doc.insert_text(0, "Intro line.\nTarget SLA: under 50 ms sync.\nFooter line.")

        sla_range = doc.create_range_for_substring("under 50 ms sync.")
        s0, e0, txt0 = doc.resolve_range(sla_range)
        self.assertEqual(txt0, "under 50 ms sync.")

        # Insert 95 chars at offset 0 (above the pinned span)
        prefix = "[Upstream RFC Header Added Concurrently by Human Engineer Before Bedrock Responds]\n"
        doc.insert_text(0, prefix)
        s1, e1, txt1 = doc.resolve_range(sla_range)
        self.assertEqual(txt1, "under 50 ms sync.")
        self.assertEqual(s1 - s0, len(prefix))
        self.assertEqual(e1 - e0, len(prefix))

        # Delete "Intro line.\n" right above the target span
        intro_idx = doc.get_text().find("Intro line.\n")
        doc.delete_range(intro_idx, intro_idx + len("Intro line.\n"))
        _, _, txt2 = doc.resolve_range(sla_range)
        self.assertEqual(txt2, "under 50 ms sync.")

    def test_04_anchor_boundary_bias_excludes_adjacent_typing(self) -> None:
        """Typing immediately before `start(before)` or immediately after `end(after)` stays outside the range."""
        doc = YataDocument(actor_id="peer-maya")
        doc.insert_text(0, "Alpha [TARGET] Omega")
        rng = doc.create_range_for_substring("[TARGET]")
        s, e, _ = doc.resolve_range(rng)

        doc.insert_text(s, "LEFT_")
        _, e_after_left, _ = doc.resolve_range(rng)
        doc.insert_text(e_after_left, "_RIGHT")

        _, _, resolved_txt = doc.resolve_range(rng)
        self.assertEqual(resolved_txt, "[TARGET]")
        self.assertEqual(doc.get_text(), "Alpha LEFT_[TARGET]_RIGHT Omega")

    def test_05_varint_and_state_vector_binary_roundtrip(self) -> None:
        """LEB128 varint and CHSV StateVector encode/decode round-trip across boundary integers."""
        samples = [0, 1, 127, 128, 255, 16384, 10599520, 2**31 - 1]
        blob = b"".join(encode_varint(x) for x in samples)
        pos = 0
        decoded = []
        for _ in samples:
            val, pos = decode_varint(blob, pos)
            decoded.append(val)
        self.assertEqual(decoded, samples)

        sv = {"bedrock-nova": 148, "peer-alex": 42, "peer-maya": 915}
        frame = encode_state_vector(sv)
        self.assertLess(len(frame), 56)
        self.assertEqual(decode_state_vector(frame), sv)

    def test_06_rle_binary_delta_compression_and_sync(self) -> None:
        """CHDL binary delta packs sequential keystrokes into RLE runs and achieves >90% byte reduction vs CRDT JSON."""
        doc_a = YataDocument(actor_id="peer-maya", doc_id="doc-wire")
        doc_a.insert_text(0, "Base specification text for collaborative workspace sync. " * 25)
        doc_b = doc_a.clone("peer-alex")

        # Peer A makes a small 48-char edit
        doc_a.insert_text(doc_a.active_item_count(), "Incremental p99 latency verified at 31.4 ms.")

        sv_b = encode_state_vector(doc_b.get_contiguous_state_vector())
        report = compute_sync_delta(doc_a, sv_b)

        self.assertEqual(report["rle_packed_records"], 1)
        self.assertGreater(report["reduction_vs_crdt_snapshot_pct"], 95.0)
        self.assertLess(report["delta_bytes"], 120)

        doc_b.apply_ops(decode_binary_delta(report["delta_raw"]))
        self.assertEqual(doc_a.get_text(), doc_b.get_text())

    def test_07_sqlite_wal_oplog_compaction_and_hydration(self) -> None:
        """SQLiteCRDTStore appends WAL frames, folds them into a compacted checkpoint, and hydrates losslessly."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteCRDTStore(db_path=Path(tmp) / "test_chorus.db")
            doc = YataDocument(actor_id="peer-maya", doc_id="doc-sqlite")

            for i in range(6):
                ops, tx = doc.insert_text(doc.active_item_count(), f"Paragraph {i}: CRDT checkpoint test.\n")
                store.append_ops(
                    doc_id=doc.doc_id,
                    actor_id="peer-maya",
                    tx_id=tx.tx_id,
                    ops=list(ops),
                    state_vector=doc.get_contiguous_state_vector(),
                )
            d_ops, d_tx = doc.delete_range(0, 12)
            store.append_ops(
                doc_id=doc.doc_id,
                actor_id="peer-maya",
                tx_id=d_tx.tx_id,
                ops=list(d_ops),
                state_vector=doc.get_contiguous_state_vector(),
            )

            stats_before = store.get_storage_stats()
            self.assertEqual(stats_before["wal_frames"], 7)

            compact = store.compact_document_log(doc)
            self.assertEqual(compact["frames_folded"], 7)
            self.assertGreater(compact["checkpoint_vs_json_reduction_pct"], 80.0)

            hydrated = store.hydrate_document("doc-sqlite")
            self.assertEqual(hydrated.get_text(), doc.get_text())
            self.assertEqual(hydrated.tombstone_count(), doc.tombstone_count())

    def test_08_selective_per_actor_undo_preserves_subsequent_peer_edits(self) -> None:
        """Undoing AI's middle rewrite restores replaced text AND preserves later edits by human peers."""
        doc = YataDocument(actor_id="peer-maya", doc_id="doc-undo")
        doc.insert_text(0, "Sec 1: Original intro. Sec 2: Draft sentence. Sec 3: End.", tx_id="tx-init")
        sec2_range = doc.create_range_for_substring("Draft sentence.")

        # 1. Maya edits Sec 1
        doc.insert_text(7, "[Maya] ", actor_id="peer-maya", author_kind="human", tx_id="tx-maya-1")
        # 2. AI replaces Sec 2
        doc.replace_anchor_range(
            sec2_range,
            "AI rewritten specification clause with verbose assumptions.",
            actor_id="bedrock-nova",
            author_kind="ai",
            tx_id="tx-nova-1",
        )
        # 3. Alex edits Sec 3 AFTER AI's edit
        doc.insert_text(
            doc.active_item_count(),
            " [Alex sign-off]",
            actor_id="peer-alex",
            author_kind="human",
            tx_id="tx-alex-1",
        )

        # Selectively undo ONLY bedrock-nova!
        muts, undo_rec = doc.undo_actor("bedrock-nova")
        self.assertIsNotNone(undo_rec)
        self.assertGreater(len(muts), 0)
        self.assertEqual(
            doc.get_text(),
            "Sec 1: [Maya] Original intro. Sec 2: Draft sentence. Sec 3: End. [Alex sign-off]",
        )

    def test_09_selective_undo_propagates_across_synced_replicas(self) -> None:
        """Inverse tombstone/restore mutations generated by undo_actor converge across remote replicas."""
        doc_a = YataDocument(actor_id="peer-a")
        doc_a.insert_text(0, "KeepA ", actor_id="peer-a", tx_id="tx-a")
        doc_a.insert_text(6, "RevertB ", actor_id="peer-b", tx_id="tx-b")
        doc_a.insert_text(14, "KeepC", actor_id="peer-c", tx_id="tx-c")

        doc_b = doc_a.clone("peer-b")
        doc_a.undo_actor("peer-b")
        res = sync_peers_binary(doc_a, doc_b)

        self.assertTrue(res["converged"])
        self.assertEqual(doc_b.get_text(), "KeepA KeepC")

    def test_10_multi_peer_convergence_fuzzer_80_seeds(self) -> None:
        """80-seed 4-peer partition, packet-shuffle, and duplicate-delivery fuzzer hits 100.0% convergence."""
        report = run_convergence_fuzz_suite(num_seeds=80, replicas_count=4, ops_per_seed=14, base_seed=42000)
        self.assertEqual(report["seeds_converged"], 80)
        self.assertEqual(report["convergence_rate_pct"], 100.0)
        self.assertEqual(report["interleaved_word_corruptions"], 0)

    def test_11_partition_simulator_modes(self) -> None:
        """All 3 UI partition simulator modes converge identically across Maya, Alex, and Nova."""
        for mode in ("split_brain_rfc", "same_offset_collision", "out_of_order_burst"):
            sim = simulate_partition_scenario(seed=10599520, scenario_mode=mode, num_fuzz_seeds=12)
            self.assertTrue(sim["all_three_identical"], f"Failed convergence in mode {mode}")
            self.assertEqual(sim["fuzz_batch"]["convergence_rate_pct"], 100.0)

    def test_12_workspace_coedit_offline_deterministic_path(self) -> None:
        """ChorusWorkspace coedit_document preserves concurrent human typing while shifting AI anchor cleanly."""
        prev = os.environ.get("CHORUS_OFFLINE_LLM")
        os.environ["CHORUS_OFFLINE_LLM"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ws = ChorusWorkspace(store=SQLiteCRDTStore(db_path=Path(tmp) / "ws.db"))
                res = ws.coedit_document(
                    doc_id="doc-rfc-sync",
                    section_key="anchor_protocol",
                    ai_mode="tighten_metrics",
                    concurrent_human_text="[Human concurrent edit above anchor]\n",
                    human_position="above_anchor",
                )
                proof = res["anchor_shift_proof"]
                self.assertGreater(proof["shifted_by_chars"], 0)
                self.assertTrue(proof["human_concurrent_preserved"])
                self.assertTrue(proof["all_replicas_converged"])
        finally:
            if prev is None:
                os.environ.pop("CHORUS_OFFLINE_LLM", None)
            else:
                os.environ["CHORUS_OFFLINE_LLM"] = prev

    def test_13_api_router_and_lambda_handler_endpoints(self) -> None:
        """Verify `/api/health`, `/api/doc/state`, `/api/sync/delta`, `/api/undo/actor`, `/api/fuzz/run`, `/api/reset`."""
        st, health = route_api_request("GET", "/api/health", {}, {})
        self.assertEqual(st, 200)
        self.assertEqual(health["status"], "ok")

        st, doc_state = route_api_request("GET", "/api/doc/state", {"doc_id": "doc-rfc-sync"}, {})
        self.assertEqual(st, 200)
        self.assertEqual(doc_state["active_doc_id"], "doc-rfc-sync")

        st, delta = route_api_request("POST", "/api/sync/delta", {}, {"doc_id": "doc-rfc-sync"})
        self.assertEqual(st, 200)
        self.assertTrue(delta["lagging_peer_converged"])
        self.assertTrue(delta["hydrated_roundtrip_verified"])

        st, staged = route_api_request("POST", "/api/undo/actor", {}, {"doc_id": "doc-undo-spec", "action": "stage"})
        self.assertEqual(st, 200)
        self.assertEqual(len(staged["staged_transactions"]), 3)

        st, undone = route_api_request(
            "POST", "/api/undo/actor", {}, {"doc_id": "doc-undo-spec", "actor_id": "bedrock-nova"}
        )
        self.assertEqual(st, 200)
        self.assertTrue(undone["undo_applied"])
        self.assertNotIn("30 seconds", undone["text"])
        self.assertIn("Maya @ 10:14:02", undone["text"])
        self.assertIn("Alex @ 10:14:05", undone["text"])

        lam_resp = lambda_handler(
            {
                "rawPath": "/api/fuzz/run",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"seed": 10599520, "num_fuzz_seeds": 15}),
            }
        )
        self.assertEqual(lam_resp["statusCode"], 200)
        parsed = json.loads(lam_resp["body"])
        self.assertTrue(parsed["all_three_identical"])

        st, rst = route_api_request("POST", "/api/reset", {}, {})
        self.assertEqual(st, 200)
        self.assertEqual(rst["status"], "reset_complete")


if __name__ == "__main__":
    unittest.main()
