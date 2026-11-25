"""Scratch Yata Sequence CRDT Engine, Logical Relative Anchors & Selective Per-Actor Causal Undo."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Immutable logical address: (actor_id, lamport_clock)
ItemId = Tuple[str, int]


@dataclass
class CRDTItem:
    """Atomic sequence CRDT element addressed by (actor_id, lamport_clock)."""

    id: ItemId
    origin_left: Optional[ItemId]
    origin_right: Optional[ItemId]
    text: str
    deleted: bool = False
    author_kind: str = "human"
    tx_id: str = "tx-init"
    actor_seq: int = 0
    # LWW causal register (op_lamport, op_actor_id, is_deleted) so out-of-order delete/undo converge
    delete_version: Tuple[int, str, bool] = (0, "", False)

    @property
    def actor_id(self) -> str:
        return self.id[0]

    @property
    def lamport_clock(self) -> int:
        return self.id[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": list(self.id),
            "origin_left": list(self.origin_left) if self.origin_left else None,
            "origin_right": list(self.origin_right) if self.origin_right else None,
            "text": self.text,
            "deleted": self.deleted,
            "author_kind": self.author_kind,
            "tx_id": self.tx_id,
            "actor_seq": self.actor_seq,
            "delete_version": [self.delete_version[0], self.delete_version[1], self.delete_version[2]],
        }


@dataclass(frozen=True)
class InsertOp:
    """Replicated character insertion carrying Yata causal left/right origins."""

    id: ItemId
    actor_seq: int
    origin_left: Optional[ItemId]
    origin_right: Optional[ItemId]
    text: str
    author_kind: str = "human"
    tx_id: str = "tx-init"

    @property
    def actor_id(self) -> str:
        return self.id[0]

    @property
    def lamport_clock(self) -> int:
        return self.id[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": "insert",
            "id": list(self.id),
            "actor_seq": self.actor_seq,
            "origin_left": list(self.origin_left) if self.origin_left else None,
            "origin_right": list(self.origin_right) if self.origin_right else None,
            "text": self.text,
            "author_kind": self.author_kind,
            "tx_id": self.tx_id,
        }


@dataclass(frozen=True)
class MutationOp:
    """Replicated tombstone delete or selective-undo restore operation."""

    op_id: ItemId
    actor_seq: int
    target_id: ItemId
    set_deleted: bool
    author_kind: str = "human"
    tx_id: str = "tx-mut"
    reason: str = "delete"

    @property
    def actor_id(self) -> str:
        return self.op_id[0]

    @property
    def lamport_clock(self) -> int:
        return self.op_id[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": "mutation",
            "op_id": list(self.op_id),
            "actor_seq": self.actor_seq,
            "target_id": list(self.target_id),
            "set_deleted": self.set_deleted,
            "author_kind": self.author_kind,
            "tx_id": self.tx_id,
            "reason": self.reason,
        }


CRDTOp = Union[InsertOp, MutationOp]


@dataclass(frozen=True)
class RelativeAnchor:
    """Logical cursor/comment pin bound to an immutable CRDT item ID rather than a brittle integer index."""

    item_id: Optional[ItemId]
    bias: str = "after"  # "before" or "after"

    def __post_init__(self) -> None:
        if self.bias not in ("before", "after"):
            raise ValueError(f"Invalid anchor bias: {self.bias}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": list(self.item_id) if self.item_id else None,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RelativeAnchor":
        raw_id = raw.get("item_id")
        item_id = (str(raw_id[0]), int(raw_id[1])) if raw_id else None
        return cls(item_id=item_id, bias=str(raw.get("bias") or "after"))


@dataclass
class RelativeRange:
    """Start/end logical anchor pair pinning an AI edit target or inline review comment."""

    start: RelativeAnchor
    end: RelativeAnchor
    label: str = ""
    owner_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "label": self.label,
            "owner_id": self.owner_id,
        }


@dataclass
class TransactionRecord:
    """Causal edit group enabling selective per-actor undo without touching later peer edits."""

    tx_id: str
    actor_id: str
    author_kind: str
    summary: str
    inserted_ids: List[ItemId] = field(default_factory=list)
    deleted_ids: List[ItemId] = field(default_factory=list)
    restored_ids: List[ItemId] = field(default_factory=list)
    undone: bool = False
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "actor_id": self.actor_id,
            "author_kind": self.author_kind,
            "summary": self.summary,
            "inserted_count": len(self.inserted_ids),
            "deleted_count": len(self.deleted_ids),
            "restored_count": len(self.restored_ids),
            "undone": self.undone,
            "timestamp_ms": self.timestamp_ms,
        }


class YataDocument:
    """Conflict-free sequence CRDT implementing Yata non-interleaving rules and relative anchors."""

    def __init__(self, actor_id: str = "peer-local", author_kind: str = "human", doc_id: str = "doc-default") -> None:
        self.doc_id = doc_id
        self.actor_id = actor_id
        self.author_kind = author_kind

        self._lamport: int = 0
        self._local_seq: int = 0

        self._seq: List[CRDTItem] = []
        self._items: Dict[ItemId, CRDTItem] = {}
        self._pos_cache: Dict[ItemId, int] = {}
        self._pos_dirty: bool = False

        # Operation history keyed by (actor_id, actor_seq) for gap-free StateVector delta sync
        self._ops_by_actor: Dict[str, Dict[int, CRDTOp]] = {}
        self._seen_op_keys: Set[Tuple[str, str, int]] = set()

        # Out-of-order causal buffers for packet reordering under network partitions
        self._pending_inserts: Dict[ItemId, InsertOp] = {}
        self._pending_mutations: Dict[ItemId, List[MutationOp]] = {}

        # Per-actor causal undo ledger & live presence cursors
        self._transactions: List[TransactionRecord] = []
        self._tx_index: Dict[str, TransactionRecord] = {}
        self._presence: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Internal indexing & clock helpers
    # ------------------------------------------------------------------
    def _ensure_pos_index(self) -> Dict[ItemId, int]:
        if self._pos_dirty:
            self._pos_cache = {item.id: idx for idx, item in enumerate(self._seq)}
            self._pos_dirty = False
        return self._pos_cache

    def _next_local_stamp(self) -> Tuple[ItemId, int]:
        self._lamport += 1
        self._local_seq += 1
        return (self.actor_id, self._lamport), self._local_seq

    def _observe_clock(self, remote_lamport: int, remote_actor: str, remote_seq: int) -> None:
        if remote_lamport > self._lamport:
            self._lamport = remote_lamport
        if remote_actor == self.actor_id and remote_seq > self._local_seq:
            self._local_seq = remote_seq

    def _record_op(self, op: CRDTOp) -> None:
        actor = op.actor_id
        bucket = self._ops_by_actor.setdefault(actor, {})
        bucket[op.actor_seq] = op

    # ------------------------------------------------------------------
    # Yata non-interleaving sequence integration & causal buffering
    # ------------------------------------------------------------------
    def _is_causally_ready(self, op: InsertOp) -> bool:
        if op.origin_left is not None and op.origin_left not in self._items:
            return False
        if op.origin_right is not None and op.origin_right not in self._items:
            return False
        return True

    def _integrate_ready_insert(self, op: InsertOp) -> None:
        if op.id in self._items:
            return

        self._observe_clock(op.lamport_clock, op.actor_id, op.actor_seq)
        seq = self._seq
        n = len(seq)

        # Fast O(1) path for sequential tail/cursor typing when origin_left and origin_right are adjacent
        if not self._pos_dirty:
            left_idx = self._pos_cache[op.origin_left] if op.origin_left is not None else -1
            right_idx = self._pos_cache[op.origin_right] if op.origin_right is not None else n
        else:
            pos = self._ensure_pos_index()
            left_idx = pos[op.origin_left] if op.origin_left is not None else -1
            right_idx = pos[op.origin_right] if op.origin_right is not None else n

        if left_idx + 1 == right_idx:
            dest_idx = right_idx
        else:
            pos = self._ensure_pos_index()
            dest_idx = left_idx + 1
            scanning = False
            for i in range(left_idx + 1, right_idx):
                other = seq[i]
                o_left = pos[other.origin_left] if other.origin_left is not None else -1
                o_right = pos[other.origin_right] if other.origin_right is not None else n

                if o_left < left_idx:
                    break
                elif o_left == left_idx:
                    if op.id > other.id:
                        scanning = False
                    elif o_right == right_idx:
                        break
                    else:
                        scanning = True

                if not scanning:
                    dest_idx = i + 1

        item = CRDTItem(
            id=op.id,
            origin_left=op.origin_left,
            origin_right=op.origin_right,
            text=op.text,
            deleted=False,
            author_kind=op.author_kind,
            tx_id=op.tx_id,
            actor_seq=op.actor_seq,
        )

        if dest_idx == n:
            seq.append(item)
            if not self._pos_dirty:
                self._pos_cache[item.id] = n
        else:
            seq.insert(dest_idx, item)
            self._pos_dirty = True

        self._items[item.id] = item
        self._record_op(op)

        # Apply any buffered mutations waiting on this item
        waiting_muts = self._pending_mutations.pop(item.id, None)
        if waiting_muts:
            for mut in waiting_muts:
                self._apply_ready_mutation(mut)

    def _apply_ready_mutation(self, op: MutationOp) -> None:
        self._observe_clock(op.lamport_clock, op.actor_id, op.actor_seq)
        target = self._items[op.target_id]
        incoming_ver = (op.lamport_clock, op.actor_id, op.set_deleted)
        cur_ver = target.delete_version
        if (incoming_ver[0], incoming_ver[1]) > (cur_ver[0], cur_ver[1]):
            target.delete_version = incoming_ver
            target.deleted = op.set_deleted
        self._record_op(op)

    def _drain_pending(self) -> None:
        while self._pending_inserts:
            progressed = False
            # Deterministic order over buffered items
            ready_ids = [
                item_id
                for item_id, pending_op in sorted(self._pending_inserts.items())
                if self._is_causally_ready(pending_op)
            ]
            for item_id in ready_ids:
                op = self._pending_inserts.pop(item_id)
                self._integrate_ready_insert(op)
                progressed = True
            if not progressed:
                break

    def apply_op(self, op: CRDTOp) -> bool:
        """Apply a local or remote CRDT operation idempotently (safe under duplicates & reordering)."""
        if isinstance(op, InsertOp):
            dedup_key = ("I", op.actor_id, op.lamport_clock)
            if dedup_key in self._seen_op_keys or op.id in self._items:
                return False
            self._seen_op_keys.add(dedup_key)
            if self._is_causally_ready(op):
                self._integrate_ready_insert(op)
                self._drain_pending()
            else:
                self._pending_inserts[op.id] = op
            return True

        if isinstance(op, MutationOp):
            dedup_key = ("M", op.actor_id, op.lamport_clock)
            if dedup_key in self._seen_op_keys:
                return False
            self._seen_op_keys.add(dedup_key)
            if op.target_id in self._items:
                self._apply_ready_mutation(op)
            else:
                self._pending_mutations.setdefault(op.target_id, []).append(op)
            return True

        raise TypeError(f"Unsupported CRDT operation: {type(op)!r}")

    def apply_ops(self, ops: List[CRDTOp]) -> int:
        applied = 0
        for op in ops:
            if self.apply_op(op):
                applied += 1
        return applied

    # ------------------------------------------------------------------
    # Visible character indexing & local edit operations
    # ------------------------------------------------------------------
    def _visible_items(self) -> List[CRDTItem]:
        return [it for it in self._seq if not it.deleted]

    def get_text(self) -> str:
        return "".join(it.text for it in self._seq if not it.deleted)

    def items(self) -> List[CRDTItem]:
        return list(self._seq)

    def pending_count(self) -> int:
        return len(self._pending_inserts) + sum(len(v) for v in self._pending_mutations.values())

    def tombstone_count(self) -> int:
        return sum(1 for it in self._seq if it.deleted)

    def active_item_count(self) -> int:
        return sum(1 for it in self._seq if not it.deleted)

    def insert_text(
        self,
        index: int,
        text: str,
        *,
        actor_id: Optional[str] = None,
        author_kind: Optional[str] = None,
        tx_id: Optional[str] = None,
        summary: str = "",
    ) -> Tuple[List[InsertOp], TransactionRecord]:
        """Insert a string at visible character offset `index`, emitting Yata InsertOps."""
        eff_actor = actor_id or self.actor_id
        eff_kind = author_kind or self.author_kind
        eff_tx = tx_id or f"tx-{eff_actor}-{uuid.uuid4().hex[:8]}"

        vis = self._visible_items()
        clamped = max(0, min(index, len(vis)))

        origin_left: Optional[ItemId] = vis[clamped - 1].id if clamped > 0 else None
        origin_right: Optional[ItemId] = vis[clamped].id if clamped < len(vis) else None

        emitted: List[InsertOp] = []
        inserted_ids: List[ItemId] = []

        for ch in text:
            if eff_actor == self.actor_id:
                item_id, seq_no = self._next_local_stamp()
            else:
                self._lamport += 1
                actor_bucket = self._ops_by_actor.get(eff_actor, {})
                seq_no = (max(actor_bucket.keys()) if actor_bucket else 0) + 1
                item_id = (eff_actor, self._lamport)

            op = InsertOp(
                id=item_id,
                actor_seq=seq_no,
                origin_left=origin_left,
                origin_right=origin_right,
                text=ch,
                author_kind=eff_kind,
                tx_id=eff_tx,
            )
            self.apply_op(op)
            emitted.append(op)
            inserted_ids.append(item_id)
            # Subsequent characters in the same typed run chain off the previous character
            origin_left = item_id

        tx = self._register_tx(
            tx_id=eff_tx,
            actor_id=eff_actor,
            author_kind=eff_kind,
            summary=summary or f"Insert {len(text)} chars at offset {clamped}",
            inserted_ids=inserted_ids,
        )
        self.update_presence(
            actor_id=eff_actor,
            author_kind=eff_kind,
            anchor=RelativeAnchor(inserted_ids[-1] if inserted_ids else origin_left, "after"),
            status="editing",
        )
        return emitted, tx

    def delete_range(
        self,
        start_index: int,
        end_index: int,
        *,
        actor_id: Optional[str] = None,
        author_kind: Optional[str] = None,
        tx_id: Optional[str] = None,
        summary: str = "",
    ) -> Tuple[List[MutationOp], TransactionRecord]:
        """Tombstone visible characters in `[start_index, end_index)`."""
        eff_actor = actor_id or self.actor_id
        eff_kind = author_kind or self.author_kind
        eff_tx = tx_id or f"tx-{eff_actor}-{uuid.uuid4().hex[:8]}"

        vis = self._visible_items()
        s = max(0, min(start_index, len(vis)))
        e = max(s, min(end_index, len(vis)))
        targets = vis[s:e]

        emitted: List[MutationOp] = []
        deleted_ids: List[ItemId] = []
        for target in targets:
            op = self._emit_mutation(
                target_id=target.id,
                set_deleted=True,
                eff_actor=eff_actor,
                eff_kind=eff_kind,
                eff_tx=eff_tx,
                reason="delete",
            )
            emitted.append(op)
            deleted_ids.append(target.id)

        tx = self._register_tx(
            tx_id=eff_tx,
            actor_id=eff_actor,
            author_kind=eff_kind,
            summary=summary or f"Delete {len(deleted_ids)} chars at [{s}:{e}]",
            deleted_ids=deleted_ids,
        )
        return emitted, tx

    def _emit_mutation(
        self,
        *,
        target_id: ItemId,
        set_deleted: bool,
        eff_actor: str,
        eff_kind: str,
        eff_tx: str,
        reason: str,
    ) -> MutationOp:
        if eff_actor == self.actor_id:
            op_id, seq_no = self._next_local_stamp()
        else:
            self._lamport += 1
            actor_bucket = self._ops_by_actor.get(eff_actor, {})
            seq_no = (max(actor_bucket.keys()) if actor_bucket else 0) + 1
            op_id = (eff_actor, self._lamport)

        op = MutationOp(
            op_id=op_id,
            actor_seq=seq_no,
            target_id=target_id,
            set_deleted=set_deleted,
            author_kind=eff_kind,
            tx_id=eff_tx,
            reason=reason,
        )
        self.apply_op(op)
        return op

    def _register_tx(
        self,
        *,
        tx_id: str,
        actor_id: str,
        author_kind: str,
        summary: str,
        inserted_ids: Optional[List[ItemId]] = None,
        deleted_ids: Optional[List[ItemId]] = None,
        restored_ids: Optional[List[ItemId]] = None,
    ) -> TransactionRecord:
        existing = self._tx_index.get(tx_id)
        if existing:
            if inserted_ids:
                existing.inserted_ids.extend(inserted_ids)
            if deleted_ids:
                existing.deleted_ids.extend(deleted_ids)
            if restored_ids:
                existing.restored_ids.extend(restored_ids)
            if summary:
                existing.summary = summary
            return existing

        tx = TransactionRecord(
            tx_id=tx_id,
            actor_id=actor_id,
            author_kind=author_kind,
            summary=summary,
            inserted_ids=list(inserted_ids or []),
            deleted_ids=list(deleted_ids or []),
            restored_ids=list(restored_ids or []),
        )
        self._transactions.append(tx)
        self._tx_index[tx_id] = tx
        return tx

    # ------------------------------------------------------------------
    # Logical Relative Anchors & Anchor-Scoped Replacements
    # ------------------------------------------------------------------
    def create_anchor(self, index: int, bias: str = "after") -> RelativeAnchor:
        """Bind a logical anchor to the CRDT item at `index` with `before` or `after` bias."""
        vis = self._visible_items()
        if not vis:
            return RelativeAnchor(item_id=None, bias=bias)
        if bias == "before":
            if index <= 0:
                return RelativeAnchor(item_id=vis[0].id, bias="before")
            if index >= len(vis):
                return RelativeAnchor(item_id=vis[-1].id, bias="after")
            return RelativeAnchor(item_id=vis[index].id, bias="before")
        else:
            if index <= 0:
                return RelativeAnchor(item_id=vis[0].id, bias="before")
            clamped = min(index - 1, len(vis) - 1)
            return RelativeAnchor(item_id=vis[clamped].id, bias="after")

    def resolve_anchor(self, anchor: RelativeAnchor) -> int:
        """Resolve a `RelativeAnchor` to its current visible character offset in O(N)."""
        if anchor.item_id is None:
            return 0 if anchor.bias == "before" else self.active_item_count()

        vis_offset = 0
        for item in self._seq:
            if item.id == anchor.item_id:
                if anchor.bias == "before":
                    return vis_offset
                return vis_offset + (len(item.text) if not item.deleted else 0)
            if not item.deleted:
                vis_offset += len(item.text)
        return vis_offset

    def create_range(
        self,
        start_index: int,
        end_index: int,
        *,
        label: str = "",
        owner_id: str = "",
    ) -> RelativeRange:
        vis = self._visible_items()
        if not vis or start_index >= end_index:
            anchor = self.create_anchor(start_index, bias="after")
            return RelativeRange(start=anchor, end=anchor, label=label, owner_id=owner_id)

        s = max(0, min(start_index, len(vis) - 1))
        e = max(s + 1, min(end_index, len(vis)))
        start_anchor = RelativeAnchor(item_id=vis[s].id, bias="before")
        end_anchor = RelativeAnchor(item_id=vis[e - 1].id, bias="after")
        return RelativeRange(start=start_anchor, end=end_anchor, label=label, owner_id=owner_id)

    def create_range_for_substring(
        self,
        target_substring: str,
        *,
        label: str = "",
        owner_id: str = "",
    ) -> RelativeRange:
        full_text = self.get_text()
        idx = full_text.find(target_substring)
        if idx < 0:
            raise ValueError(f"Substring not found in document: {target_substring[:48]!r}")
        return self.create_range(idx, idx + len(target_substring), label=label, owner_id=owner_id)

    def resolve_range(self, rel_range: RelativeRange) -> Tuple[int, int, str]:
        s = self.resolve_anchor(rel_range.start)
        e = self.resolve_anchor(rel_range.end)
        if e < s:
            s, e = e, s
        txt = self.get_text()
        return s, e, txt[s:e]

    def replace_anchor_range(
        self,
        rel_range: RelativeRange,
        replacement_text: str,
        *,
        actor_id: Optional[str] = None,
        author_kind: Optional[str] = None,
        tx_id: Optional[str] = None,
        summary: str = "",
    ) -> Tuple[List[CRDTOp], TransactionRecord, Tuple[int, int]]:
        """Atomically tombstone the current resolved span of `rel_range` and insert `replacement_text`."""
        eff_actor = actor_id or self.actor_id
        eff_kind = author_kind or self.author_kind
        eff_tx = tx_id or f"tx-{eff_actor}-{uuid.uuid4().hex[:8]}"

        s, e, _ = self.resolve_range(rel_range)
        all_ops: List[CRDTOp] = []

        # Capture boundary neighbors BEFORE tombstoning so new text anchors at the exact logical slot
        vis = self._visible_items()
        origin_left: Optional[ItemId] = vis[s - 1].id if s > 0 and s <= len(vis) else None
        origin_right: Optional[ItemId] = vis[e].id if e < len(vis) else None

        deleted_ids: List[ItemId] = []
        for target in vis[s:e]:
            mut_op = self._emit_mutation(
                target_id=target.id,
                set_deleted=True,
                eff_actor=eff_actor,
                eff_kind=eff_kind,
                eff_tx=eff_tx,
                reason="replace_delete",
            )
            all_ops.append(mut_op)
            deleted_ids.append(target.id)

        inserted_ids: List[ItemId] = []
        cur_left = origin_left
        for ch in replacement_text:
            if eff_actor == self.actor_id:
                item_id, seq_no = self._next_local_stamp()
            else:
                self._lamport += 1
                actor_bucket = self._ops_by_actor.get(eff_actor, {})
                seq_no = (max(actor_bucket.keys()) if actor_bucket else 0) + 1
                item_id = (eff_actor, self._lamport)

            ins_op = InsertOp(
                id=item_id,
                actor_seq=seq_no,
                origin_left=cur_left,
                origin_right=origin_right,
                text=ch,
                author_kind=eff_kind,
                tx_id=eff_tx,
            )
            self.apply_op(ins_op)
            all_ops.append(ins_op)
            inserted_ids.append(item_id)
            cur_left = item_id

        tx = self._register_tx(
            tx_id=eff_tx,
            actor_id=eff_actor,
            author_kind=eff_kind,
            summary=summary or f"Rewrite anchored range ({len(deleted_ids)} -> {len(inserted_ids)} chars)",
            inserted_ids=inserted_ids,
            deleted_ids=deleted_ids,
        )
        new_s = self.resolve_anchor(RelativeAnchor(inserted_ids[0], "before")) if inserted_ids else s
        new_e = self.resolve_anchor(RelativeAnchor(inserted_ids[-1], "after")) if inserted_ids else new_s
        self.update_presence(
            actor_id=eff_actor,
            author_kind=eff_kind,
            anchor=RelativeAnchor(inserted_ids[-1] if inserted_ids else origin_left, "after"),
            selection_start=RelativeAnchor(inserted_ids[0], "before") if inserted_ids else None,
            status="idle",
        )
        return all_ops, tx, (new_s, new_e)

    # ------------------------------------------------------------------
    # Selective Per-Actor Causal Undo
    # ------------------------------------------------------------------
    def undo_actor(
        self,
        target_actor_id: str,
        *,
        tx_id: Optional[str] = None,
    ) -> Tuple[List[MutationOp], Optional[TransactionRecord]]:
        """
        Invert ONLY `target_actor_id`'s latest (or specified `tx_id`) transaction via CRDT tombstones
        and restores, preserving every interleaved character typed by other human or AI peers.
        """
        target_tx: Optional[TransactionRecord] = None
        if tx_id:
            cand = self._tx_index.get(tx_id)
            if cand and cand.actor_id == target_actor_id and not cand.undone:
                target_tx = cand
        else:
            for cand in reversed(self._transactions):
                if cand.actor_id == target_actor_id and not cand.undone:
                    target_tx = cand
                    break

        # Fallback if remote ops arrived over wire without local TransactionRecord metadata:
        # group items belonging to target_actor_id's latest non-initial tx_id
        if target_tx is None:
            seen_tx_ids: List[str] = []
            for it in self._seq:
                if it.actor_id == target_actor_id and it.tx_id != "tx-init" and not it.deleted:
                    if tx_id is None or it.tx_id == tx_id:
                        if it.tx_id not in seen_tx_ids:
                            seen_tx_ids.append(it.tx_id)
            if not seen_tx_ids:
                return [], None
            chosen_tx_id = seen_tx_ids[-1]
            ins_ids = [it.id for it in self._seq if it.actor_id == target_actor_id and it.tx_id == chosen_tx_id]
            target_tx = self._register_tx(
                tx_id=chosen_tx_id,
                actor_id=target_actor_id,
                author_kind=self._items[ins_ids[0]].author_kind if ins_ids else "human",
                summary=f"Remote transaction {chosen_tx_id}",
                inserted_ids=ins_ids,
            )

        undo_tx_id = f"undo-{target_tx.tx_id}-{uuid.uuid4().hex[:6]}"
        emitted: List[MutationOp] = []
        tombstoned_now: List[ItemId] = []
        restored_now: List[ItemId] = []

        # 1. Tombstone all items inserted by target_tx
        for item_id in target_tx.inserted_ids:
            item = self._items.get(item_id)
            if item and not item.deleted:
                op = self._emit_mutation(
                    target_id=item_id,
                    set_deleted=True,
                    eff_actor=self.actor_id,
                    eff_kind=self.author_kind,
                    eff_tx=undo_tx_id,
                    reason=f"undo_insert:{target_actor_id}",
                )
                emitted.append(op)
                tombstoned_now.append(item_id)

        # 2. Restore all items that were deleted by target_tx
        for item_id in target_tx.deleted_ids:
            item = self._items.get(item_id)
            if item and item.deleted:
                op = self._emit_mutation(
                    target_id=item_id,
                    set_deleted=False,
                    eff_actor=self.actor_id,
                    eff_kind=self.author_kind,
                    eff_tx=undo_tx_id,
                    reason=f"undo_restore:{target_actor_id}",
                )
                emitted.append(op)
                restored_now.append(item_id)

        target_tx.undone = True
        undo_record = TransactionRecord(
            tx_id=undo_tx_id,
            actor_id=self.actor_id,
            author_kind=self.author_kind,
            summary=f"Selective undo of {target_actor_id} ({target_tx.tx_id})",
            deleted_ids=tombstoned_now,
            restored_ids=restored_now,
            undone=True,
        )
        self._transactions.append(undo_record)
        self._tx_index[undo_tx_id] = undo_record
        return emitted, undo_record

    # ------------------------------------------------------------------
    # StateVector, Attribution Spans, Presence & Cloning
    # ------------------------------------------------------------------
    def get_contiguous_state_vector(self) -> Dict[str, int]:
        """Return `actor_id -> max_contiguous_actor_seq` across all integrated operations."""
        sv: Dict[str, int] = {}
        for actor, seq_map in self._ops_by_actor.items():
            contig = 0
            while (contig + 1) in seq_map:
                contig += 1
            if contig > 0:
                sv[actor] = contig
        return sv

    def export_ops_since(self, remote_sv: Dict[str, int]) -> List[CRDTOp]:
        """Return minimal causal operation list missing from `remote_sv`."""
        missing: List[CRDTOp] = []
        for actor in sorted(self._ops_by_actor.keys()):
            known_seq = int(remote_sv.get(actor, 0))
            seq_map = self._ops_by_actor[actor]
            for seq_no in sorted(seq_map.keys()):
                if seq_no > known_seq:
                    missing.append(seq_map[seq_no])
        # Sort Inserts in causal Lamport order before Mutations so peers integrate with minimal buffering
        inserts = sorted([op for op in missing if isinstance(op, InsertOp)], key=lambda o: (o.lamport_clock, o.actor_id))
        muts = sorted([op for op in missing if isinstance(op, MutationOp)], key=lambda o: (o.lamport_clock, o.actor_id))
        return inserts + muts

    def export_all_ops(self) -> List[CRDTOp]:
        return self.export_ops_since({})

    def clone(self, new_actor_id: str, new_author_kind: str = "human") -> "YataDocument":
        """Fork an independent peer replica initialized with the current integrated operation history."""
        fork = YataDocument(actor_id=new_actor_id, author_kind=new_author_kind, doc_id=self.doc_id)
        fork.apply_ops(self.export_all_ops())
        for tx in self._transactions:
            copied = TransactionRecord(
                tx_id=tx.tx_id,
                actor_id=tx.actor_id,
                author_kind=tx.author_kind,
                summary=tx.summary,
                inserted_ids=list(tx.inserted_ids),
                deleted_ids=list(tx.deleted_ids),
                restored_ids=list(tx.restored_ids),
                undone=tx.undone,
                timestamp_ms=tx.timestamp_ms,
            )
            fork._transactions.append(copied)
            fork._tx_index[copied.tx_id] = copied
        return fork

    def update_presence(
        self,
        *,
        actor_id: str,
        author_kind: str,
        anchor: RelativeAnchor,
        selection_start: Optional[RelativeAnchor] = None,
        status: str = "online",
    ) -> None:
        self._presence[actor_id] = {
            "actor_id": actor_id,
            "author_kind": author_kind,
            "anchor": anchor,
            "selection_start": selection_start,
            "status": status,
            "updated_at_ms": int(time.time() * 1000),
        }

    def get_presence_cursors(self) -> List[Dict[str, Any]]:
        cursors: List[Dict[str, Any]] = []
        for actor_id, info in sorted(self._presence.items()):
            head_idx = self.resolve_anchor(info["anchor"])
            sel_start_idx = (
                self.resolve_anchor(info["selection_start"])
                if info.get("selection_start") is not None
                else head_idx
            )
            cursors.append(
                {
                    "actor_id": actor_id,
                    "author_kind": info["author_kind"],
                    "cursor_offset": head_idx,
                    "selection_start_offset": min(sel_start_idx, head_idx),
                    "selection_end_offset": max(sel_start_idx, head_idx),
                    "anchor": info["anchor"].to_dict(),
                    "status": info["status"],
                }
            )
        return cursors

    def get_author_spans(self) -> List[Dict[str, Any]]:
        """Coalesce contiguous visible characters by `(actor_id, author_kind, tx_id)` for live UI attribution."""
        spans: List[Dict[str, Any]] = []
        for item in self._seq:
            if item.deleted:
                continue
            if (
                spans
                and spans[-1]["actor_id"] == item.actor_id
                and spans[-1]["author_kind"] == item.author_kind
                and spans[-1]["tx_id"] == item.tx_id
            ):
                spans[-1]["text"] += item.text
            else:
                spans.append(
                    {
                        "actor_id": item.actor_id,
                        "author_kind": item.author_kind,
                        "tx_id": item.tx_id,
                        "text": item.text,
                    }
                )
        return spans

    def list_transactions(self) -> List[Dict[str, Any]]:
        return [tx.to_dict() for tx in reversed(self._transactions) if tx.tx_id != "tx-init"]

    def to_full_snapshot_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "actor_id": self.actor_id,
            "lamport_clock": self._lamport,
            "text": self.get_text(),
            "state_vector": self.get_contiguous_state_vector(),
            "items": [it.to_dict() for it in self._seq],
            "transactions": self.list_transactions(),
        }
