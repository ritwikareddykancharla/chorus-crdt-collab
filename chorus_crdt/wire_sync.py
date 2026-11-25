"""Compact Binary StateVector & RLE Delta Sync Codec (`CHSV` / `CHDL`) + SQLite WAL Compaction."""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .yata_engine import CRDTOp, InsertOp, ItemId, MutationOp, YataDocument

STATE_VECTOR_MAGIC = b"CHSV\x01"
BINARY_DELTA_MAGIC = b"CHDL\x01"

OP_TAG_INSERT_RUN = 0x01
OP_TAG_MUTATION = 0x02

AUTHOR_KIND_TO_CODE = {"system": 0, "human": 1, "ai": 2}
CODE_TO_AUTHOR_KIND = {0: "system", 1: "human", 2: "ai"}


# ----------------------------------------------------------------------
# Unsigned LEB128 Varint & Length-Prefixed UTF-8 Primitives
# ----------------------------------------------------------------------
def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"Varint requires non-negative integer, got {value}")
    out = bytearray()
    v = value
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def decode_varint(buf: bytes, offset: int = 0) -> Tuple[int, int]:
    result = 0
    shift = 0
    pos = offset
    n = len(buf)
    while pos < n:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("Varint overflow (>64 bits)")
    raise ValueError("Truncated varint frame")


def _encode_str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return encode_varint(len(raw)) + raw


def _decode_str(buf: bytes, offset: int) -> Tuple[str, int]:
    length, pos = decode_varint(buf, offset)
    end = pos + length
    if end > len(buf):
        raise ValueError("Truncated UTF-8 string in binary frame")
    return buf[pos:end].decode("utf-8"), end


# ----------------------------------------------------------------------
# Binary StateVector Codec (`actor_id -> max_contiguous_seq`)
# ----------------------------------------------------------------------
def encode_state_vector(sv: Dict[str, int]) -> bytes:
    """Encode a `StateVector` map into a compact varint `CHSV\\x01` binary frame."""
    out = bytearray(STATE_VECTOR_MAGIC)
    entries = [(actor, clock) for actor, clock in sorted(sv.items()) if clock > 0]
    out.extend(encode_varint(len(entries)))
    for actor, clock in entries:
        out.extend(_encode_str(actor))
        out.extend(encode_varint(int(clock)))
    return bytes(out)


def decode_state_vector(frame: bytes) -> Dict[str, int]:
    """Decode a `CHSV\\x01` binary frame into `actor_id -> max_contiguous_seq`."""
    if not frame.startswith(STATE_VECTOR_MAGIC):
        raise ValueError("Invalid StateVector magic header")
    pos = len(STATE_VECTOR_MAGIC)
    count, pos = decode_varint(frame, pos)
    sv: Dict[str, int] = {}
    for _ in range(count):
        actor, pos = _decode_str(frame, pos)
        clock, pos = decode_varint(frame, pos)
        sv[actor] = clock
    return sv


# ----------------------------------------------------------------------
# Run-Length Block (RLE) Binary Delta Codec (`CHDL\x01`)
# ----------------------------------------------------------------------
def _pack_insert_runs(ops: List[CRDTOp]) -> List[Tuple[str, Any]]:
    """
    Coalesce contiguous sequential character InsertOps from the same actor/tx into single RLE runs.
    """
    records: List[Tuple[str, Any]] = []
    i = 0
    n = len(ops)
    while i < n:
        op = ops[i]
        if isinstance(op, InsertOp):
            chars = [op.text]
            prev_id = op.id
            prev_lamport = op.lamport_clock
            prev_seq = op.actor_seq
            j = i + 1
            while j < n:
                nxt = ops[j]
                if (
                    isinstance(nxt, InsertOp)
                    and nxt.actor_id == op.actor_id
                    and nxt.author_kind == op.author_kind
                    and nxt.tx_id == op.tx_id
                    and nxt.lamport_clock == prev_lamport + 1
                    and nxt.actor_seq == prev_seq + 1
                    and nxt.origin_left == prev_id
                    and nxt.origin_right == op.origin_right
                    and len(nxt.text) == 1
                ):
                    chars.append(nxt.text)
                    prev_id = nxt.id
                    prev_lamport = nxt.lamport_clock
                    prev_seq = nxt.actor_seq
                    j += 1
                else:
                    break
            records.append(("insert_run", (op, "".join(chars))))
            i = j
        else:
            records.append(("mutation", op))
            i += 1
    return records


def _encode_opt_item_id(item_id: Optional[ItemId], actor_to_idx: Dict[str, int]) -> bytes:
    if item_id is None:
        return b"\x00"
    out = bytearray(b"\x01")
    out.extend(encode_varint(actor_to_idx[item_id[0]]))
    out.extend(encode_varint(item_id[1]))
    return bytes(out)


def _decode_opt_item_id(buf: bytes, pos: int, actors: List[str]) -> Tuple[Optional[ItemId], int]:
    present = buf[pos]
    pos += 1
    if present == 0:
        return None, pos
    a_idx, pos = decode_varint(buf, pos)
    clock, pos = decode_varint(buf, pos)
    return (actors[a_idx], clock), pos


def encode_binary_delta(ops: List[CRDTOp]) -> Tuple[bytes, int]:
    """
    Encode CRDT operations into a compact `CHDL\\x01` binary delta frame using:
      1. Deduplicated actor string table + tx_id string table
      2. Run-length block compression (RLE) for sequential character runs
      3. Unsigned LEB128 varint clocks
    Returns `(binary_bytes, rle_record_count)`.
    """
    actor_set = set()
    tx_set = set()
    for op in ops:
        actor_set.add(op.actor_id)
        tx_set.add(op.tx_id)
        if isinstance(op, InsertOp):
            if op.origin_left:
                actor_set.add(op.origin_left[0])
            if op.origin_right:
                actor_set.add(op.origin_right[0])
        elif isinstance(op, MutationOp):
            actor_set.add(op.target_id[0])

    actors = sorted(actor_set)
    actor_to_idx = {a: idx for idx, a in enumerate(actors)}
    txs = sorted(tx_set)
    tx_to_idx = {t: idx for idx, t in enumerate(txs)}

    packed = _pack_insert_runs(ops)

    out = bytearray(BINARY_DELTA_MAGIC)
    out.extend(encode_varint(len(actors)))
    for a in actors:
        out.extend(_encode_str(a))

    out.extend(encode_varint(len(txs)))
    for t in txs:
        out.extend(_encode_str(t))

    out.extend(encode_varint(len(packed)))
    for kind, payload in packed:
        if kind == "insert_run":
            head_op, run_text = payload
            out.append(OP_TAG_INSERT_RUN)
            out.extend(encode_varint(actor_to_idx[head_op.actor_id]))
            out.extend(encode_varint(head_op.lamport_clock))
            out.extend(encode_varint(head_op.actor_seq))
            out.append(AUTHOR_KIND_TO_CODE.get(head_op.author_kind, 1))
            out.extend(encode_varint(tx_to_idx[head_op.tx_id]))
            out.extend(_encode_opt_item_id(head_op.origin_left, actor_to_idx))
            out.extend(_encode_opt_item_id(head_op.origin_right, actor_to_idx))
            out.extend(_encode_str(run_text))
        else:
            mut: MutationOp = payload
            flags = (0x01 if mut.set_deleted else 0x00) | (AUTHOR_KIND_TO_CODE.get(mut.author_kind, 1) << 1)
            out.append(OP_TAG_MUTATION)
            out.append(flags)
            out.extend(encode_varint(actor_to_idx[mut.actor_id]))
            out.extend(encode_varint(mut.lamport_clock))
            out.extend(encode_varint(mut.actor_seq))
            out.extend(encode_varint(tx_to_idx[mut.tx_id]))
            out.extend(encode_varint(actor_to_idx[mut.target_id[0]]))
            out.extend(encode_varint(mut.target_id[1]))
            out.extend(_encode_str(mut.reason))

    return bytes(out), len(packed)


def decode_binary_delta(frame: bytes) -> List[CRDTOp]:
    """Decode a `CHDL\\x01` binary delta frame back into atomic `InsertOp` and `MutationOp` records."""
    if not frame.startswith(BINARY_DELTA_MAGIC):
        raise ValueError("Invalid BinaryDelta magic header")
    pos = len(BINARY_DELTA_MAGIC)

    actor_count, pos = decode_varint(frame, pos)
    actors: List[str] = []
    for _ in range(actor_count):
        a, pos = _decode_str(frame, pos)
        actors.append(a)

    tx_count, pos = decode_varint(frame, pos)
    txs: List[str] = []
    for _ in range(tx_count):
        t, pos = _decode_str(frame, pos)
        txs.append(t)

    rec_count, pos = decode_varint(frame, pos)
    ops: List[CRDTOp] = []

    for _ in range(rec_count):
        tag = frame[pos]
        pos += 1
        if tag == OP_TAG_INSERT_RUN:
            a_idx, pos = decode_varint(frame, pos)
            start_lamport, pos = decode_varint(frame, pos)
            start_seq, pos = decode_varint(frame, pos)
            kind_code = frame[pos]
            pos += 1
            tx_idx, pos = decode_varint(frame, pos)
            origin_left, pos = _decode_opt_item_id(frame, pos, actors)
            origin_right, pos = _decode_opt_item_id(frame, pos, actors)
            run_text, pos = _decode_str(frame, pos)

            actor_id = actors[a_idx]
            author_kind = CODE_TO_AUTHOR_KIND.get(kind_code, "human")
            tx_id = txs[tx_idx]
            cur_left = origin_left

            for offset, ch in enumerate(run_text):
                item_id = (actor_id, start_lamport + offset)
                ops.append(
                    InsertOp(
                        id=item_id,
                        actor_seq=start_seq + offset,
                        origin_left=cur_left,
                        origin_right=origin_right,
                        text=ch,
                        author_kind=author_kind,
                        tx_id=tx_id,
                    )
                )
                cur_left = item_id
        elif tag == OP_TAG_MUTATION:
            flags = frame[pos]
            pos += 1
            set_deleted = bool(flags & 0x01)
            author_kind = CODE_TO_AUTHOR_KIND.get((flags >> 1) & 0x03, "human")
            a_idx, pos = decode_varint(frame, pos)
            lamport, pos = decode_varint(frame, pos)
            seq_no, pos = decode_varint(frame, pos)
            tx_idx, pos = decode_varint(frame, pos)
            tgt_a_idx, pos = decode_varint(frame, pos)
            tgt_clock, pos = decode_varint(frame, pos)
            reason, pos = _decode_str(frame, pos)
            ops.append(
                MutationOp(
                    op_id=(actors[a_idx], lamport),
                    actor_seq=seq_no,
                    target_id=(actors[tgt_a_idx], tgt_clock),
                    set_deleted=set_deleted,
                    author_kind=author_kind,
                    tx_id=txs[tx_idx],
                    reason=reason,
                )
            )
        else:
            raise ValueError(f"Unknown binary delta record tag: {tag:#x}")

    return ops


def compute_sync_delta(source_doc: YataDocument, target_sv_bytes: bytes) -> Dict[str, Any]:
    """Compute minimal binary delta from `source_doc` for a peer advertising `target_sv_bytes`."""
    t0 = time.perf_counter()
    remote_sv = decode_state_vector(target_sv_bytes)
    missing_ops = source_doc.export_ops_since(remote_sv)
    delta_bytes, rle_runs = encode_binary_delta(missing_ops)
    encode_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    t1 = time.perf_counter()
    _ = decode_binary_delta(delta_bytes)
    decode_ms = round((time.perf_counter() - t1) * 1000.0, 4)

    full_crdt_json = json.dumps(source_doc.to_full_snapshot_dict(), separators=(",", ":")).encode("utf-8")
    plain_doc_json = json.dumps(
        {
            "doc_id": source_doc.doc_id,
            "text": source_doc.get_text(),
            "state_vector": source_doc.get_contiguous_state_vector(),
            "author_spans": source_doc.get_author_spans(),
        },
        separators=(",", ":"),
    ).encode("utf-8")

    wire_total = len(target_sv_bytes) + len(delta_bytes)
    vs_crdt_pct = round((1.0 - (wire_total / max(1, len(full_crdt_json)))) * 100.0, 2)
    vs_plain_pct = round((1.0 - (wire_total / max(1, len(plain_doc_json)))) * 100.0, 2)

    return {
        "state_vector_bytes": len(target_sv_bytes),
        "delta_bytes": len(delta_bytes),
        "wire_exchange_bytes": wire_total,
        "missing_atomic_ops": len(missing_ops),
        "rle_packed_records": rle_runs,
        "full_crdt_snapshot_bytes": len(full_crdt_json),
        "plain_doc_json_bytes": len(plain_doc_json),
        "reduction_vs_crdt_snapshot_pct": vs_crdt_pct,
        "reduction_vs_plain_json_pct": vs_plain_pct,
        "compression_ratio_vs_crdt": round(len(full_crdt_json) / max(1, wire_total), 1),
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "state_vector_hex": target_sv_bytes.hex(),
        "delta_hex_preview": delta_bytes[:96].hex(),
        "delta_raw": delta_bytes,
    }


def sync_peers_binary(doc_a: YataDocument, doc_b: YataDocument) -> Dict[str, Any]:
    """Perform a full bidirectional binary StateVector + CHDL delta exchange between `doc_a` and `doc_b`."""
    sv_a = encode_state_vector(doc_a.get_contiguous_state_vector())
    sv_b = encode_state_vector(doc_b.get_contiguous_state_vector())

    a_to_b = compute_sync_delta(doc_a, sv_b)
    b_to_a = compute_sync_delta(doc_b, sv_a)

    ops_for_b = decode_binary_delta(a_to_b["delta_raw"])
    ops_for_a = decode_binary_delta(b_to_a["delta_raw"])

    applied_b = doc_b.apply_ops(ops_for_b)
    applied_a = doc_a.apply_ops(ops_for_a)

    return {
        "converged": doc_a.get_text() == doc_b.get_text(),
        "a_to_b_bytes": a_to_b["delta_bytes"],
        "b_to_a_bytes": b_to_a["delta_bytes"],
        "applied_on_b": applied_b,
        "applied_on_a": applied_a,
        "sv_a_bytes": len(sv_a),
        "sv_b_bytes": len(sv_b),
    }


# ----------------------------------------------------------------------
# SQLite WAL Incremental Op-Log & Checkpoint Compaction Store
# ----------------------------------------------------------------------
class SQLiteCRDTStore:
    """Persistent SQLite WAL operation log and RLE checkpoint compactor."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        base_dir = Path("/tmp/chorus-crdt-runtime")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (base_dir / "chorus_oplog.db")
        self._init_schema()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS crdt_oplog (
                    seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    tx_id TEXT NOT NULL,
                    atomic_ops INTEGER NOT NULL,
                    rle_records INTEGER NOT NULL,
                    byte_size INTEGER NOT NULL,
                    delta_blob BLOB NOT NULL,
                    state_vector_blob BLOB NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS crdt_checkpoints (
                    doc_id TEXT PRIMARY KEY,
                    compacted_blob BLOB NOT NULL,
                    state_vector_blob BLOB NOT NULL,
                    active_chars INTEGER NOT NULL,
                    tombstones_retained INTEGER NOT NULL,
                    oplog_frames_folded INTEGER NOT NULL,
                    raw_oplog_bytes INTEGER NOT NULL,
                    compacted_bytes INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                """
            )

    def append_ops(
        self,
        *,
        doc_id: str,
        actor_id: str,
        tx_id: str,
        ops: List[CRDTOp],
        state_vector: Dict[str, int],
    ) -> Dict[str, Any]:
        delta_blob, rle_count = encode_binary_delta(ops)
        sv_blob = encode_state_vector(state_vector)
        now_ms = int(time.time() * 1000)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO crdt_oplog (
                    doc_id, actor_id, tx_id, atomic_ops, rle_records,
                    byte_size, delta_blob, state_vector_blob, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, actor_id, tx_id, len(ops), rle_count, len(delta_blob), delta_blob, sv_blob, now_ms),
            )
            row_id = int(cur.lastrowid or 0)
        return {
            "seq_id": row_id,
            "doc_id": doc_id,
            "atomic_ops": len(ops),
            "rle_records": rle_count,
            "delta_bytes": len(delta_blob),
            "sv_bytes": len(sv_blob),
        }

    def compact_document_log(self, doc: YataDocument) -> Dict[str, Any]:
        """
        Fold incremental SQLite WAL op-log frames for `doc.doc_id` into a single RLE-compacted binary
        checkpoint and purge superseded op-log rows while verifying lossless hydration.
        """
        t0 = time.perf_counter()
        all_ops = doc.export_all_ops()
        compacted_blob, rle_records = encode_binary_delta(all_ops)
        sv_blob = encode_state_vector(doc.get_contiguous_state_vector())
        now_ms = int(time.time() * 1000)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(byte_size + LENGTH(state_vector_blob)), 0) FROM crdt_oplog WHERE doc_id = ?",
                (doc.doc_id,),
            ).fetchone()
            frames_count = int(row[0] or 0)
            raw_oplog_bytes = int(row[1] or 0)

            prev_cp = conn.execute(
                "SELECT compacted_bytes FROM crdt_checkpoints WHERE doc_id = ?",
                (doc.doc_id,),
            ).fetchone()
            prior_total = raw_oplog_bytes + (int(prev_cp[0]) if prev_cp else 0)

            conn.execute(
                """
                INSERT INTO crdt_checkpoints (
                    doc_id, compacted_blob, state_vector_blob, active_chars,
                    tombstones_retained, oplog_frames_folded, raw_oplog_bytes,
                    compacted_bytes, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    compacted_blob=excluded.compacted_blob,
                    state_vector_blob=excluded.state_vector_blob,
                    active_chars=excluded.active_chars,
                    tombstones_retained=excluded.tombstones_retained,
                    oplog_frames_folded=crdt_checkpoints.oplog_frames_folded + excluded.oplog_frames_folded,
                    raw_oplog_bytes=excluded.raw_oplog_bytes,
                    compacted_bytes=excluded.compacted_bytes,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (
                    doc.doc_id,
                    compacted_blob,
                    sv_blob,
                    doc.active_item_count(),
                    doc.tombstone_count(),
                    frames_count,
                    prior_total,
                    len(compacted_blob),
                    now_ms,
                ),
            )
            conn.execute("DELETE FROM crdt_oplog WHERE doc_id = ?", (doc.doc_id,))

        compact_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        full_json_bytes = len(json.dumps(doc.to_full_snapshot_dict(), separators=(",", ":")).encode("utf-8"))
        return {
            "doc_id": doc.doc_id,
            "frames_folded": frames_count,
            "rle_records": rle_records,
            "prior_wal_bytes": prior_total,
            "compacted_checkpoint_bytes": len(compacted_blob),
            "full_json_snapshot_bytes": full_json_bytes,
            "checkpoint_vs_json_reduction_pct": round((1.0 - len(compacted_blob) / max(1, full_json_bytes)) * 100.0, 2),
            "active_chars": doc.active_item_count(),
            "tombstones_retained": doc.tombstone_count(),
            "compact_ms": compact_ms,
        }

    def hydrate_document(self, doc_id: str, actor_id: str = "hydrated-peer") -> YataDocument:
        """Reconstruct a `YataDocument` from its SQLite compacted checkpoint + any trailing WAL frames."""
        doc = YataDocument(actor_id=actor_id, doc_id=doc_id)
        with self._connect() as conn:
            cp = conn.execute(
                "SELECT compacted_blob FROM crdt_checkpoints WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if cp and cp[0]:
                doc.apply_ops(decode_binary_delta(bytes(cp[0])))

            rows = conn.execute(
                "SELECT delta_blob FROM crdt_oplog WHERE doc_id = ? ORDER BY seq_id ASC",
                (doc_id,),
            ).fetchall()
            for (blob,) in rows:
                doc.apply_ops(decode_binary_delta(bytes(blob)))
        return doc

    def get_storage_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            oplog_row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(byte_size), 0) FROM crdt_oplog"
            ).fetchone()
            cp_row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(compacted_bytes), 0), COALESCE(SUM(oplog_frames_folded), 0) FROM crdt_checkpoints"
            ).fetchone()
        return {
            "db_path": str(self.db_path),
            "wal_frames": int(oplog_row[0] or 0),
            "wal_bytes": int(oplog_row[1] or 0),
            "checkpoints_count": int(cp_row[0] or 0),
            "checkpoints_bytes": int(cp_row[1] or 0),
            "total_frames_compacted": int(cp_row[2] or 0),
        }

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM crdt_oplog")
            conn.execute("DELETE FROM crdt_checkpoints")
