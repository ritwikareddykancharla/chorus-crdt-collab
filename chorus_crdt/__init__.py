"""ChorusCRDT: Multiplayer Human + AI Sequence CRDT, Binary Delta Sync & Selective Causal Undo."""

from .yata_engine import (
    CRDTItem,
    InsertOp,
    ItemId,
    MutationOp,
    RelativeAnchor,
    RelativeRange,
    TransactionRecord,
    YataDocument,
)
from .wire_sync import (
    SQLiteCRDTStore,
    compute_sync_delta,
    decode_binary_delta,
    decode_state_vector,
    encode_binary_delta,
    encode_state_vector,
    encode_varint,
    decode_varint,
    sync_peers_binary,
)
from .fuzzer import run_convergence_fuzz_suite, simulate_partition_scenario
from .ai_collaborator import ChorusWorkspace, invoke_bedrock_nova

__all__ = [
    "CRDTItem",
    "InsertOp",
    "ItemId",
    "MutationOp",
    "RelativeAnchor",
    "RelativeRange",
    "TransactionRecord",
    "YataDocument",
    "SQLiteCRDTStore",
    "compute_sync_delta",
    "decode_binary_delta",
    "decode_state_vector",
    "encode_binary_delta",
    "encode_state_vector",
    "encode_varint",
    "decode_varint",
    "sync_peers_binary",
    "run_convergence_fuzz_suite",
    "simulate_partition_scenario",
    "ChorusWorkspace",
    "invoke_bedrock_nova",
]
