"""Multi-Peer Network Partition, Packet Reordering & Convergence Fuzzer for ChorusCRDT."""
from __future__ import annotations

import random
import statistics
import time
from typing import Any, Dict, List, Tuple

from .wire_sync import sync_peers_binary
from .yata_engine import CRDTOp, RelativeRange, YataDocument

ATOMIC_TOKENS = [
    "[RFC-Sync]",
    "[Nova-Peer]",
    "[SLA<50ms]",
    "[CursorPin]",
    "[DeltaBin]",
    "[UndoSafe]",
    "[ZeroLock]",
    "[QuorumOK]",
]

REPLICA_PROFILES: List[Tuple[str, str]] = [
    ("peer-maya", "human"),
    ("peer-alex", "human"),
    ("bedrock-nova", "ai"),
    ("peer-sam", "human"),
]


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _safe_boundary_offsets(txt: str) -> List[int]:
    """Return character offsets outside `<...>` atomic word markers so peers don't deliberately type inside an already-received word."""
    valid: List[int] = []
    depth = 0
    for idx, ch in enumerate(txt):
        if depth == 0:
            valid.append(idx)
        if ch == "<":
            depth += 1
        elif ch == ">" and depth > 0:
            depth -= 1
    if depth == 0:
        valid.append(len(txt))
    return valid or [0]


def run_single_fuzz_seed(
    seed: int,
    *,
    replicas_count: int = 4,
    ops_per_seed: int = 16,
) -> Dict[str, Any]:
    """
    Execute one deterministic multi-replica chaos test:
      - Phase 1: Concurrent multi-char word insertions + deletions with shuffled & duplicated packets
      - Phase 2: Split-brain network partition ([0, 2] vs [1, 3]) + selective per-actor undo
      - Phase 3: Chaotic out-of-order replay with 15% packet drop + binary StateVector delta heal
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    chosen_profiles = REPLICA_PROFILES[: max(3, min(replicas_count, len(REPLICA_PROFILES)))]

    # Seed base document on replica 0 and clone across all peers
    origin = YataDocument(actor_id="doc-seed", author_kind="system", doc_id=f"fuzz-{seed}")
    base_text = "Section 1: Shared CRDT state. Section 2: Anchor target span. Section 3: Tail notes."
    origin.insert_text(0, base_text, tx_id="tx-init", summary="Seed base RFC")
    pinned_range: RelativeRange = origin.create_range_for_substring("Anchor target span", label="ai-target")

    peers: List[YataDocument] = [
        origin.clone(new_actor_id=actor, new_author_kind=kind) for actor, kind in chosen_profiles
    ]

    # Track untouched word tokens inserted with unique markers so we can prove 0 character interleaving
    injected_tokens: List[str] = []
    in_flight_packets: List[Tuple[int, CRDTOp]] = []  # (recipient_idx, op)
    duplicate_replays = 0
    reordered_deliveries = 0
    dropped_packets = 0

    # Phase 1: Online concurrent edits with packet reordering & 20% duplicates
    half_ops = max(4, ops_per_seed // 2)
    for step in range(half_ops):
        p_idx = rng.randrange(len(peers))
        peer = peers[p_idx]
        txt = peer.get_text()
        boundaries = _safe_boundary_offsets(txt)
        action = rng.random()

        if action < 0.72 or len(boundaries) < 8:
            token = f"<{rng.choice(ATOMIC_TOKENS)}#{seed % 97}_{step}>"
            pos = rng.choice(boundaries)
            emitted, _ = peer.insert_text(pos, token)
            injected_tokens.append(token)
        else:
            # Delete only contiguous non-marker base characters
            b_idx = rng.randrange(len(boundaries) - 1)
            s = boundaries[b_idx]
            e = min(len(txt), s + rng.randint(1, 4))
            if "<" not in txt[s:e] and ">" not in txt[s:e] and s in boundaries and e in boundaries:
                emitted, _ = peer.delete_range(s, e)
            else:
                token = f"<{rng.choice(ATOMIC_TOKENS)}#{seed % 97}_{step}>"
                emitted, _ = peer.insert_text(boundaries[0], token)
                injected_tokens.append(token)

        for target_idx in range(len(peers)):
            if target_idx == p_idx:
                continue
            for op in emitted:
                in_flight_packets.append((target_idx, op))
                if rng.random() < 0.20:
                    in_flight_packets.append((target_idx, op))
                    duplicate_replays += 1

    rng.shuffle(in_flight_packets)
    reordered_deliveries += len(in_flight_packets)
    for target_idx, op in in_flight_packets:
        peers[target_idx].apply_op(op)
    in_flight_packets.clear()

    # Phase 2: Split-brain network partition (Group A: even indices, Group B: odd indices)
    partition_packets: List[Tuple[int, CRDTOp]] = []
    for step in range(half_ops, ops_per_seed):
        p_idx = rng.randrange(len(peers))
        peer = peers[p_idx]
        group = p_idx % 2

        if step == ops_per_seed - 1 and peer.list_transactions():
            # Trigger a selective per-actor undo inside one partition before healing!
            emitted_muts, _ = peer.undo_actor(peer.actor_id)
            emitted_ops: List[CRDTOp] = list(emitted_muts)
            injected_tokens = [tok for tok in injected_tokens if tok in peer.get_text()]
        else:
            token = f"<{rng.choice(ATOMIC_TOKENS)}#P{group}_{step}>"
            boundaries = _safe_boundary_offsets(peer.get_text())
            pos = rng.choice(boundaries)
            emitted_ins, _ = peer.insert_text(pos, token)
            emitted_ops = list(emitted_ins)
            injected_tokens.append(token)

        for target_idx in range(len(peers)):
            if target_idx == p_idx:
                continue
            if (target_idx % 2) == group:
                for op in emitted_ops:
                    peers[target_idx].apply_op(op)
            else:
                # Cross-partition blocked; 15% dropped completely to force StateVector delta recovery
                for op in emitted_ops:
                    if rng.random() < 0.15:
                        dropped_packets += 1
                    else:
                        partition_packets.append((target_idx, op))

    diverged_during_partition = len({p.get_text() for p in peers}) > 1

    # Phase 3: Heal partition — deliver remaining shuffled packets + forward/backward StateVector sync sweep
    rng.shuffle(partition_packets)
    reordered_deliveries += len(partition_packets)
    for target_idx, op in partition_packets:
        peers[target_idx].apply_op(op)

    sync_wire_bytes = 0
    for idx in range(len(peers) - 1):
        stats = sync_peers_binary(peers[idx], peers[idx + 1])
        sync_wire_bytes += stats["a_to_b_bytes"] + stats["b_to_a_bytes"]
    for idx in reversed(range(len(peers) - 1)):
        stats = sync_peers_binary(peers[idx], peers[idx + 1])
        sync_wire_bytes += stats["a_to_b_bytes"] + stats["b_to_a_bytes"]

    texts = [p.get_text() for p in peers]
    text_converged = len(set(texts)) == 1

    id_sequences = [tuple((it.id, it.deleted) for it in p.items()) for p in peers]
    seq_converged = len(set(id_sequences)) == 1

    anchor_spans = [p.resolve_range(pinned_range) for p in peers]
    anchors_converged = len(set(anchor_spans)) == 1

    # Verify zero interleaved word corruptions on surviving tokens
    final_text = texts[0]
    interleaved_corruptions = 0
    for tok in injected_tokens:
        # If the token's characters were not tombstoned by selective undo, the entire `<...>` string must be contiguous
        # Check if first char of unique suffix `#...>` is present in final_text
        suffix = tok.split("#", 1)[1]
        if suffix in final_text and tok not in final_text:
            interleaved_corruptions += 1

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "seed": seed,
        "converged": bool(text_converged and seq_converged and anchors_converged and interleaved_corruptions == 0),
        "text_converged": text_converged,
        "seq_converged": seq_converged,
        "anchors_converged": anchors_converged,
        "diverged_during_partition": diverged_during_partition,
        "interleaved_corruptions": interleaved_corruptions,
        "duplicate_replays": duplicate_replays,
        "reordered_deliveries": reordered_deliveries,
        "dropped_packets_healed": dropped_packets,
        "sync_wire_bytes": sync_wire_bytes,
        "final_char_length": len(final_text),
        "elapsed_ms": elapsed_ms,
    }


def run_convergence_fuzz_suite(
    *,
    num_seeds: int = 320,
    replicas_count: int = 4,
    ops_per_seed: int = 16,
    base_seed: int = 10599520,
) -> Dict[str, Any]:
    """Run `num_seeds` deterministic multi-peer partition & out-of-order fuzz seeds."""
    t0 = time.perf_counter()
    seed_results: List[Dict[str, Any]] = []
    for offset in range(num_seeds):
        res = run_single_fuzz_seed(
            base_seed + offset,
            replicas_count=replicas_count,
            ops_per_seed=ops_per_seed,
        )
        seed_results.append(res)

    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    durations = sorted(r["elapsed_ms"] for r in seed_results)
    converged_count = sum(1 for r in seed_results if r["converged"])
    total_corruptions = sum(r["interleaved_corruptions"] for r in seed_results)
    total_dups = sum(r["duplicate_replays"] for r in seed_results)
    total_reordered = sum(r["reordered_deliveries"] for r in seed_results)
    total_dropped_healed = sum(r["dropped_packets_healed"] for r in seed_results)

    return {
        "seeds_tested": num_seeds,
        "replicas_per_seed": replicas_count,
        "ops_per_seed": ops_per_seed,
        "base_seed": base_seed,
        "seeds_converged": converged_count,
        "convergence_rate_pct": round((converged_count / max(1, num_seeds)) * 100.0, 2),
        "interleaved_word_corruptions": total_corruptions,
        "duplicate_packets_replayed": total_dups,
        "reordered_packets_delivered": total_reordered,
        "dropped_packets_recovered_by_delta": total_dropped_healed,
        "seed_p50_ms": round(_percentile(durations, 50.0), 3),
        "seed_p95_ms": round(_percentile(durations, 95.0), 3),
        "mean_seed_ms": round(statistics.mean(durations), 3) if durations else 0.0,
        "total_suite_ms": total_ms,
        "sample_seeds": seed_results[:8],
    }


def simulate_partition_scenario(
    *,
    seed: int = 10599520,
    scenario_mode: str = "split_brain_rfc",
    num_fuzz_seeds: int = 60,
) -> Dict[str, Any]:
    """
    Run an inspectable 3-peer (`@maya-eng`, `@alex-sre`, `@nova-editor`) network partition trace
    plus a live batch of `num_fuzz_seeds` randomized convergence seeds for UI Tab 02.
    """
    base_doc = YataDocument(actor_id="system-seed", author_kind="system", doc_id="sim-partition")
    initial = (
        "## Sync Protocol SLA\n"
        "Target fan-out latency: <50 ms across active document editors.\n"
        "Wire format: compact StateVector frames with causal anchors."
    )
    base_doc.insert_text(0, initial, tx_id="tx-init")
    anchor = base_doc.create_range_for_substring("<50 ms across active document editors.", label="sla-pin")

    maya = base_doc.clone("peer-maya", "human")
    alex = base_doc.clone("peer-alex", "human")
    nova = base_doc.clone("bedrock-nova", "ai")

    if scenario_mode == "same_offset_collision":
        # All 3 peers insert different multi-word phrases at the exact same character index concurrently!
        idx = initial.find("<50 ms")
        maya_ops, _ = maya.insert_text(idx, "[Maya: p99 verified in us-west-2] ", summary="Maya concurrent insert")
        alex_ops, _ = alex.insert_text(idx, "[Alex: 200-editor fan-out] ", summary="Alex concurrent insert")
        nova_ops, _ = nova.insert_text(idx, "[Nova: zero central lock] ", summary="Nova concurrent insert")
    elif scenario_mode == "out_of_order_burst":
        maya_ops, _ = maya.insert_text(0, "[Header: RFC-042 Approved] ", summary="Maya top banner")
        m2, _ = maya.insert_text(len(maya.get_text()), "\n[Footer: Signed off by Maya]", summary="Maya footer")
        maya_ops.extend(m2)
        alex_ops, _ = alex.insert_text(initial.find("Wire format:"), "[Transport: WebSocket + HTTP/3 fallback]\n", summary="Alex transport note")
        nova_ops, _, _ = nova.replace_anchor_range(
            anchor,
            "<50 ms p99 across 250 simultaneous human and AI peer cursors.",
            summary="Nova tighten SLA target",
        )
    else:
        # Default split_brain_rfc: Partition A (Maya + Nova) vs Partition B (Alex offline on airplane Wi-Fi)
        maya_ops, _ = maya.insert_text(0, "[Status: Draft Ready for Design Review]\n", summary="Maya status header")
        nova_ops, _, _ = nova.replace_anchor_range(
            anchor,
            "under 42 ms p99 with RLE binary deltas (~$0.68/builder/year).",
            summary="Nova tighten SLA + unit economics",
        )
        alex_ops, _ = alex.insert_text(
            len(alex.get_text()),
            "\nFallback: offline edits buffer in IndexedDB and sync via CHSV StateVector.",
            summary="Alex offline partition note",
        )

    partition_views = {
        "peer_maya_before_heal": maya.get_text(),
        "peer_alex_before_heal": alex.get_text(),
        "peer_nova_before_heal": nova.get_text(),
    }

    # Deliver operations in deliberately reversed / shuffled order + duplicate replay
    rng = random.Random(seed)
    all_remote = list(maya_ops) + list(alex_ops) + list(nova_ops)
    shuffled_for_maya = list(all_remote)
    shuffled_for_alex = list(reversed(all_remote))
    rng.shuffle(shuffled_for_maya)

    maya.apply_ops(shuffled_for_maya + shuffled_for_maya[:3])  # includes duplicate replay
    alex.apply_ops(shuffled_for_alex)
    sync_ab = sync_peers_binary(maya, alex)
    sync_bc = sync_peers_binary(alex, nova)
    sync_ca = sync_peers_binary(nova, maya)

    s_idx, e_idx, pinned_after = maya.resolve_range(anchor)
    batch = run_convergence_fuzz_suite(
        num_seeds=max(10, min(num_fuzz_seeds, 400)),
        replicas_count=4,
        ops_per_seed=14,
        base_seed=seed,
    )

    return {
        "scenario_mode": scenario_mode,
        "seed": seed,
        "partition_views": partition_views,
        "converged_text": maya.get_text(),
        "all_three_identical": maya.get_text() == alex.get_text() == nova.get_text(),
        "author_spans": maya.get_author_spans(),
        "resolved_anchor": {
            "start": s_idx,
            "end": e_idx,
            "text": pinned_after,
        },
        "wire_sync_bytes": sync_ab["a_to_b_bytes"] + sync_bc["a_to_b_bytes"] + sync_ca["a_to_b_bytes"],
        "fuzz_batch": batch,
    }
