#!/usr/bin/env python3
"""
Real experiment pipeline for paper reproduction on RuleTaker and ProofWriter.

This script is designed for submission-grade experiments:
- Loads benchmark data from HuggingFace (not synthetic generation)
- Parses natural-language rule contexts into explicit schemas
- Applies controlled noise to real schemas (random / clustered)
- Trains count-based reliability estimator
- Runs three real intervention variants in a forward-chaining reasoner
- Builds proof graphs and computes cascade rate by chain tracing
- Runs degraded-estimator ablation (label noise)

No result calibration against target tables is performed.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional dependencies. We import lazily/flexibly so code review is possible before env setup.
try:
    from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset

    DATASETS_AVAILABLE = True
except Exception:
    DATASETS_AVAILABLE = False
    Dataset = object  # type: ignore
    DatasetDict = object  # type: ignore

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False


FAMILIES: Tuple[str, ...] = ("IMPLICATION", "TRANSITIVE", "DEFAULT")
MODELS: Tuple[str, ...] = (
    "instance_weighted",
    "aspic_v1_terminal",
    "aspic_v2",
    "aspic_v2_persistent",
    "schema_gating",
)
CONDITIONS: Tuple[str, ...] = (
    "clean",
    "random_negate",
    "clustered_negate",
    "random_redirect",
    "clustered_redirect",
)
PAPER_CONFIG_PATH = Path("configs/config_both_modes.json")
PAPER_MAIN_OUTDIR = "outputs_main"
PAPER_SWEEP_OUTDIR = "outputs_sweep"
PAPER_TAU_OUTDIR = "outputs_tau_sweep"
PAPER_SWEEP_RATES = [0.05, 0.10, 0.20, 0.40, 0.60]
PAPER_TAU_VALUES = [0.3, 0.5, 0.7]
RELATIONAL_VERBS: Tuple[str, ...] = (
    "like",
    "chase",
    "see",
    "visit",
    "need",
    "eat",
    "fear",
    "help",
)
VARIABLE_TOKEN = "var_x"
GENERIC_ENTITY_TERMS = {
    "someone",
    "somebody",
    "something",
    "person",
    "thing",
    "they",
    "them",
    "it",
    "he",
    "she",
}
ANAPHORA_SUBJECT_TOKENS = {"it", "they", "them", "he", "she"}


# -----------------------------
# Data classes
# -----------------------------


@dataclasses.dataclass
class Schema:
    schema_id: str
    text: str
    antecedents: Tuple[str, ...]
    consequent: str
    family: str
    source_id: str
    corrupted: bool = False


@dataclasses.dataclass
class Problem:
    problem_id: str
    depth: int
    question: str
    answer: int
    facts: Tuple[str, ...]
    schemas: List[Schema]


@dataclasses.dataclass
class SchemaRecord:
    schema_id: str
    text: str
    family: str
    source_id: str
    label: int  # 1 clean/reliable, 0 corrupted/unreliable


@dataclasses.dataclass
class EstimatorModel:
    threshold: float
    backend: str
    source_scores: Dict[str, float]
    family_scores: Dict[str, float]
    global_score: float
    metrics: Dict[str, float]

    def score(self, schema: Schema) -> float:
        if schema.source_id in self.source_scores:
            return self.source_scores[schema.source_id]
        if schema.family in self.family_scores:
            return self.family_scores[schema.family]
        return self.global_score


@dataclasses.dataclass
class DerivationNode:
    node_id: str
    proposition: str
    parent_ids: Tuple[str, ...]
    rule_id: Optional[str]
    family: Optional[str]
    source_id: Optional[str]
    reliability: float
    clarity: float
    confidence: float
    corrupted: bool
    step: int
    committed: bool


@dataclasses.dataclass
class InferenceResult:
    predicted_label: int
    predicted_node_id: Optional[str]
    nodes: Dict[str, DerivationNode]
    best_committed: Dict[str, str]
    removed_schema_ids: List[str]
    used_default_decision: bool
    transient_blocked_total: int
    transient_consumed_total: int


# -----------------------------
# Utility
# -----------------------------


def deterministic_seed(*parts: object) -> int:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) & 0x7FFFFFFF


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_articles(phrase: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", phrase.strip())


def _normalize_entity(phrase: str) -> str:
    phrase = normalize_text(phrase)
    phrase = re.sub(r"^[\W_]+|[\W_]+$", "", phrase).strip()
    phrase = _strip_articles(phrase)
    if phrase in GENERIC_ENTITY_TERMS:
        return VARIABLE_TOKEN
    return phrase


def _normalize_predicate_phrase(phrase: str) -> str:
    phrase = normalize_text(phrase)
    phrase = re.sub(r"^[\W_]+|[\W_]+$", "", phrase).strip()
    phrase = _strip_articles(phrase)
    return phrase


def _atom_unary(subj: str, pred: str, negated: bool = False) -> str:
    if not subj or not pred:
        return ""
    base = f"u|{subj}|{pred}"
    return f"not|{base}" if negated else base


def _atom_binary(subj: str, verb: str, obj: str, negated: bool = False) -> str:
    if not subj or not verb or not obj:
        return ""
    base = f"r|{subj}|{verb}|{obj}"
    return f"not|{base}" if negated else base


def normalize_prop(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    text = text.replace("?", "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    m = re.match(r"^(.+?)\s+(is|are)\s+not\s+(.+)$", text)
    if m:
        subj = _normalize_entity(m.group(1))
        pred = _normalize_predicate_phrase(m.group(3))
        atom = _atom_unary(subj, pred, negated=True)
        if atom:
            return atom

    m = re.match(r"^(.+?)\s+(is|are)\s+(.+)$", text)
    if m:
        subj = _normalize_entity(m.group(1))
        pred = _normalize_predicate_phrase(m.group(3))
        atom = _atom_unary(subj, pred, negated=False)
        if atom:
            return atom

    m = re.match(r"^(.+?)\s+does not\s+([a-z]+)\s+(.+)$", text)
    if m:
        subj = _normalize_entity(m.group(1))
        verb = m.group(2).strip().lower()
        obj = _normalize_entity(m.group(3))
        if verb in RELATIONAL_VERBS:
            atom = _atom_binary(subj, verb, obj, negated=True)
            if atom:
                return atom

    m = re.match(r"^(.+?)\s+do not\s+([a-z]+)\s+(.+)$", text)
    if m:
        subj = _normalize_entity(m.group(1))
        verb = m.group(2).strip().lower()
        obj = _normalize_entity(m.group(3))
        if verb in RELATIONAL_VERBS:
            atom = _atom_binary(subj, verb, obj, negated=True)
            if atom:
                return atom

    tokens = text.split()
    for i, tok in enumerate(tokens):
        if i == 0 or i == len(tokens) - 1:
            continue
        verb_raw = tok.strip().lower()
        verb = verb_raw[:-1] if verb_raw.endswith("s") else verb_raw
        if verb not in RELATIONAL_VERBS:
            continue
        subj = _normalize_entity(" ".join(tokens[:i]))
        obj = _normalize_entity(" ".join(tokens[i + 1 :]))
        atom = _atom_binary(subj, verb, obj, negated=False)
        if atom:
            return atom

    return text


def negate_prop(prop: str) -> str:
    prop = normalize_prop(prop)
    if prop.startswith("not|"):
        return prop[4:]
    return f"not|{prop}"


def prop_has_variable(prop: str) -> bool:
    return VARIABLE_TOKEN in prop


def substitute_variable(prop: str, entity: str) -> str:
    return prop.replace(VARIABLE_TOKEN, entity)


def prop_subject(prop: str) -> str:
    core = prop[4:] if prop.startswith("not|") else prop
    parts = core.split("|")
    if parts and parts[0] == "u" and len(parts) == 3:
        return parts[1]
    if parts and parts[0] == "r" and len(parts) == 4:
        return parts[1]
    return ""


def prop_entities(prop: str) -> List[str]:
    core = prop[4:] if prop.startswith("not|") else prop
    parts = core.split("|")
    out: List[str] = []
    if parts and parts[0] == "u" and len(parts) == 3:
        if parts[1]:
            out.append(parts[1])
    elif parts and parts[0] == "r" and len(parts) == 4:
        if parts[1]:
            out.append(parts[1])
        if parts[3]:
            out.append(parts[3])
    return [x for x in out if x and x != VARIABLE_TOKEN and not x.startswith("?")]


def extract_binding_from_pattern(pattern: str, grounded: str) -> Optional[str]:
    p = pattern.split("|")
    g = grounded.split("|")
    if len(p) != len(g):
        return None

    binding: Optional[str] = None
    for pp, gg in zip(p, g):
        if pp == VARIABLE_TOKEN:
            if binding is None:
                binding = gg
            elif binding != gg:
                return None
            continue
        if pp != gg:
            return None
    return binding


def candidate_bindings_for_pattern(pattern: str, facts: Iterable[str]) -> List[str]:
    bindings = {
        b
        for fact in facts
        for b in [extract_binding_from_pattern(pattern, fact)]
        if b
    }
    return sorted(bindings)


def run_cmd(args: Sequence[str]) -> Tuple[bool, str]:
    try:
        out = subprocess.check_output(list(args), stderr=subprocess.STDOUT, text=True)
        return True, out.strip()
    except Exception as exc:
        return False, str(exc)


def detect_hardware() -> Dict[str, object]:
    mem_total_gib = None
    mem_available_gib = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        info: Dict[str, str] = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
        if "MemTotal" in info:
            mem_total_kib = float(info["MemTotal"].split()[0])
            mem_total_gib = round(mem_total_kib / (1024.0 * 1024.0), 2)
        if "MemAvailable" in info:
            mem_av_kib = float(info["MemAvailable"].split()[0])
            mem_available_gib = round(mem_av_kib / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    disk = shutil.disk_usage(".")
    gpus: List[Dict[str, str]] = []
    ok, out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,cuda_version",
            "--format=csv,noheader",
        ]
    )
    if ok and out:
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 4:
                gpus.append(
                    {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                        "cuda_version": parts[3],
                    }
                )
    else:
        gpus.append({"error": out})

    return {
        "timestamp_epoch": int(time.time()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_logical_count": os.cpu_count(),
        "memory_total_gib": mem_total_gib,
        "memory_available_gib": mem_available_gib,
        "disk_total_gib": round(disk.total / (1024.0**3), 2),
        "disk_free_gib": round(disk.free / (1024.0**3), 2),
        "gpus": gpus,
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def configure_hf_cache_root() -> None:
    # Keep HF cache under workspace by default to avoid permission issues
    # when a global cache is owned by another user.
    if "HF_HOME" not in os.environ:
        hf_home = (Path.cwd() / ".hf_home").resolve()
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
    else:
        hf_home = Path(str(os.environ["HF_HOME"])).resolve()
        hf_home.mkdir(parents=True, exist_ok=True)

    if "HF_DATASETS_CACHE" not in os.environ:
        ds_cache = (hf_home / "datasets").resolve()
        ds_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(ds_cache)

# -----------------------------
# Rule parsing
# -----------------------------


def split_sentences(text: str) -> List[str]:
    raw = re.split(r"[\n\.]+", text)
    out: List[str] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(sent\d+|s\d+|rule\d+|fact\d+)\s*[:\-]\s*", "", line, flags=re.I)
        line = re.sub(r"^\d+\s*[:\-]\s*", "", line)
        if line:
            out.append(line)
    return out


def _replace_leading_pronoun(clause: str, subject: str) -> str:
    if not clause or not subject:
        return clause
    tokens = normalize_text(clause).split()
    if not tokens:
        return clause
    if tokens[0] in ANAPHORA_SUBJECT_TOKENS:
        return f"{subject} {' '.join(tokens[1:])}".strip()
    return clause


def _split_antecedents(text: str) -> Tuple[str, ...]:
    text = normalize_text(text.replace(" unless ", " and not "))
    chunks = [normalize_text(x) for x in re.split(r"\band\b|,", text)]
    chunks = [c for c in chunks if c]

    ants: List[str] = []
    last_subject = ""
    for chunk in chunks:
        expanded = chunk
        if last_subject:
            expanded = _replace_leading_pronoun(expanded, last_subject)

        has_copula = bool(re.search(r"\b(is|are)\b", expanded))
        has_neg_rel = bool(re.search(r"\b(does not|do not)\b", expanded))
        has_rel = any(
            re.search(rf"\b{re.escape(v)}s?\b", expanded) for v in RELATIONAL_VERBS
        )

        if not has_copula and not has_neg_rel and not has_rel and last_subject:
            expanded = f"{last_subject} is {chunk}"

        ant = normalize_prop(expanded)
        if not ant:
            continue
        ants.append(ant)

        subj = prop_subject(ant)
        if subj:
            last_subject = subj

    return tuple(ants)


def parse_rule_sentence(sentence: str) -> Optional[Tuple[Tuple[str, ...], str, str]]:
    s = normalize_text(sentence).strip(" .")
    if not s:
        return None

    antecedent_text = ""
    consequent_text = ""

    # Canonical forms:
    #   if A then B
    #   B if A
    if s.startswith("if ") and " then " in s:
        m = re.match(r"if (.+?) then (.+)$", s)
        if m:
            antecedent_text = m.group(1).strip()
            consequent_text = m.group(2).strip()
    elif " if " in s:
        m = re.match(r"(.+?) if (.+)$", s)
        if m:
            consequent_text = m.group(1).strip()
            antecedent_text = m.group(2).strip()

    # Additional common natural-language surface forms seen in RuleTaker-style data.
    if not antecedent_text or not consequent_text:
        patterns = (
            (r"^all (.+?) things are (.+)$", True),
            (r"^all (.+?) people are (.+)$", True),
            (r"^every (.+?) thing is (.+)$", True),
            (r"^every (.+?) person is (.+)$", True),
            (r"^(.+?) things are (.+)$", True),
            (r"^(.+?) people are (.+)$", True),
            (r"^all (.+?) are (.+)$", True),
            (r"^every (.+?) is (.+)$", True),
            (r"^(.+?) are always (.+)$", False),
            (r"^(.+?) is always (.+)$", False),
            (r"^(.+?) implies (.+)$", False),
        )
        for pat, is_universal_unary in patterns:
            m = re.match(pat, s)
            if m:
                lhs = m.group(1).strip()
                rhs = m.group(2).strip()
                if is_universal_unary:
                    lhs = re.sub(r"\b(things|people|thing|person)\b$", "", lhs).strip()
                    antecedent_text = f"something is {lhs}"
                    consequent_text = f"something is {rhs}"
                else:
                    antecedent_text = lhs
                    consequent_text = rhs
                break

    if not antecedent_text or not consequent_text:
        return None

    ants = _split_antecedents(antecedent_text)
    if not ants:
        return None

    anchor_subject = prop_subject(ants[0]) if ants else ""
    consequent_surface = _replace_leading_pronoun(consequent_text, anchor_subject)
    consequent = normalize_prop(consequent_surface)
    if not consequent:
        return None

    default_markers = ("unless", "typically", "normally", "usually", "except")
    if any(tok in s for tok in default_markers):
        family = "DEFAULT"
    elif len(ants) > 1:
        family = "TRANSITIVE"
    else:
        family = "IMPLICATION"
    return ants, consequent, family


def parse_context_to_facts_and_rules(context: str, example_key: str) -> Tuple[List[str], List[Schema]]:
    facts: List[str] = []
    rules: List[Schema] = []
    sentences = split_sentences(context)

    temp_rules: List[Tuple[str, Tuple[str, ...], str, str]] = []
    for idx, sentence in enumerate(sentences):
        parsed = parse_rule_sentence(sentence)
        if parsed is None:
            fact = normalize_prop(sentence)
            if fact:
                facts.append(fact)
            continue

        ants, consequent, family = parsed
        rule_text = normalize_text(sentence)
        temp_rules.append((rule_text, ants, consequent, family))

    # Promote implication to transitive if its consequent appears in another rule antecedent.
    antecedent_pool = {ant for _, ants, _, _ in temp_rules for ant in ants}

    for idx, (rule_text, ants, consequent, family) in enumerate(temp_rules):
        inferred_family = family
        if family == "IMPLICATION" and consequent in antecedent_pool:
            inferred_family = "TRANSITIVE"

        source_key = f"{inferred_family}|{rule_text}"
        source_id = f"{inferred_family}_SRC_{stable_hash(source_key, 8)}"
        schema_id = f"{example_key}_R{idx:02d}_{stable_hash(rule_text, 6)}"

        rules.append(
            Schema(
                schema_id=schema_id,
                text=rule_text,
                antecedents=ants,
                consequent=consequent,
                family=inferred_family,
                source_id=source_id,
                corrupted=False,
            )
        )

    dedup_facts = sorted(set(facts))
    return dedup_facts, rules


# -----------------------------
# RuleTaker loading
# -----------------------------


def parse_answer(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        if int(value) in (0, 1):
            return int(value)
    if isinstance(value, str):
        v = normalize_text(value)
        if v in {
            "true",
            "yes",
            "entails",
            "entailed",
            "entailment",
            "1",
            "support",
            "supports",
        }:
            return 1
        if v in {
            "false",
            "no",
            "not entails",
            "not_entails",
            "not entailment",
            "not_entailment",
            "non-entailment",
            "0",
            "refute",
            "refutes",
            "contradiction",
        }:
            return 0
    return None


def extract_text_fields(obj: object) -> List[str]:
    out: List[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj)
        return out
    if isinstance(obj, list):
        for item in obj:
            out.extend(extract_text_fields(item))
        return out
    if isinstance(obj, dict):
        for _, value in obj.items():
            out.extend(extract_text_fields(value))
    return out


def pick_first(d: Dict[str, object], keys: Sequence[str]) -> Optional[object]:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def extract_questions_from_row(row: Dict[str, object]) -> List[Tuple[str, int]]:
    questions: List[Tuple[str, int]] = []

    if "questions" in row and isinstance(row["questions"], list):
        for q in row["questions"]:
            if not isinstance(q, dict):
                continue
            q_text_obj = pick_first(q, ["question", "query", "text", "statement"])
            ans_obj = pick_first(q, ["answer", "label", "gold", "target"])
            if not isinstance(q_text_obj, str):
                continue
            ans = parse_answer(ans_obj)
            if ans is None:
                continue
            questions.append((q_text_obj, ans))
        if questions:
            return questions

    q_text_obj = pick_first(row, ["question", "query", "hypothesis", "statement"])
    ans_obj = pick_first(row, ["answer", "label", "gold", "target"])
    if isinstance(q_text_obj, str):
        ans = parse_answer(ans_obj)
        if ans is not None:
            questions.append((q_text_obj, ans))
    return questions


def extract_context_from_row(row: Dict[str, object]) -> str:
    ctx_obj = pick_first(
        row,
        [
            "context",
            "theory",
            "story",
            "facts",
            "rules",
            "background",
            "knowledge",
        ],
    )
    if ctx_obj is None:
        texts = extract_text_fields(row)
        return "\n".join(texts)

    if isinstance(ctx_obj, str):
        return ctx_obj
    if isinstance(ctx_obj, list):
        return "\n".join(extract_text_fields(ctx_obj))
    if isinstance(ctx_obj, dict):
        return "\n".join(extract_text_fields(ctx_obj))
    return ""


def select_depth_configs(dataset_name: str, depths: Sequence[int]) -> Dict[int, Optional[str]]:
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets package is required to load RuleTaker.")

    configs = get_dataset_config_names(dataset_name)
    lowered = [c.lower() for c in configs]

    selected: Dict[int, Optional[str]] = {}
    for depth in depths:
        matches = []
        pat = re.compile(rf"(?:depth|d)[-_]?{depth}(?:\b|[_-])")
        for cfg, low in zip(configs, lowered):
            if pat.search(low):
                matches.append(cfg)
        if not matches:
            selected[depth] = None
            continue

        # Prefer natural-language variants if available.
        matches.sort(key=lambda c: ("natlang" not in c.lower(), len(c), c))
        selected[depth] = matches[0]
    return selected


def load_any_split(dataset_name: str, config_name: Optional[str], split_pref: Sequence[str]):
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets package is required to load RuleTaker.")

    if config_name is None:
        dset = load_dataset(dataset_name)
    else:
        dset = load_dataset(dataset_name, config_name)

    if isinstance(dset, Dataset):
        return dset
    if isinstance(dset, DatasetDict):
        for split in split_pref:
            if split in dset:
                return dset[split]
        # fallback: first split
        first_split = next(iter(dset.keys()))
        return dset[first_split]
    raise RuntimeError("Unsupported dataset object from load_dataset.")


def flatten_ruletaker_rows(
    rows: Iterable[Dict[str, object]],
    depth: int,
    max_rules_per_problem: int,
    key_prefix: str,
    depth_filter_token: Optional[str] = None,
) -> List[Problem]:
    out: List[Problem] = []
    row_idx = 0
    for row in rows:
        row_idx += 1
        if depth_filter_token:
            cfg_val = normalize_text(str(row.get("config", "")))
            if cfg_val != normalize_text(depth_filter_token):
                continue
        ex_key = f"{key_prefix}_row{row_idx:06d}"
        context_text = extract_context_from_row(row)
        if not context_text.strip():
            continue

        facts, schemas = parse_context_to_facts_and_rules(context_text, ex_key)
        if not facts or not schemas:
            continue

        if len(schemas) > max_rules_per_problem:
            schemas = schemas[:max_rules_per_problem]

        questions = extract_questions_from_row(row)
        if not questions:
            continue

        for q_idx, (question, answer) in enumerate(questions):
            q_norm = normalize_prop(question)
            if not q_norm:
                continue
            q_neg = negate_prop(q_norm)
            if q_norm in facts or q_neg in facts:
                # Skip direct-fact questions; keep reasoning-dependent ones.
                continue
            pid = f"{ex_key}_q{q_idx:02d}"
            out.append(
                Problem(
                    problem_id=pid,
                    depth=depth,
                    question=q_norm,
                    answer=answer,
                    facts=tuple(facts),
                    schemas=copy.deepcopy(schemas),
                )
            )
    return out


def flatten_ruletaker_rows_limited(
    rows: Iterable[Dict[str, object]],
    depth: int,
    max_rules_per_problem: int,
    key_prefix: str,
    target_problems: int,
    max_rows_scan: int,
    depth_filter_token: Optional[str] = None,
) -> List[Problem]:
    out: List[Problem] = []
    row_idx = 0
    matched_rows = 0
    for row in rows:
        row_idx += 1
        if depth_filter_token:
            cfg_val = normalize_text(str(row.get("config", "")))
            if cfg_val != normalize_text(depth_filter_token):
                continue
        matched_rows += 1
        if matched_rows > max_rows_scan:
            break

        ex_key = f"{key_prefix}_row{row_idx:06d}"
        context_text = extract_context_from_row(row)
        if not context_text.strip():
            continue

        facts, schemas = parse_context_to_facts_and_rules(context_text, ex_key)
        if not facts or not schemas:
            continue

        if len(schemas) > max_rules_per_problem:
            schemas = schemas[:max_rules_per_problem]

        questions = extract_questions_from_row(row)
        if not questions:
            continue

        for q_idx, (question, answer) in enumerate(questions):
            q_norm = normalize_prop(question)
            if not q_norm:
                continue
            q_neg = negate_prop(q_norm)
            if q_norm in facts or q_neg in facts:
                # Skip direct-fact questions; keep reasoning-dependent ones.
                continue
            pid = f"{ex_key}_q{q_idx:02d}"
            out.append(
                Problem(
                    problem_id=pid,
                    depth=depth,
                    question=q_norm,
                    answer=answer,
                    facts=tuple(facts),
                    schemas=copy.deepcopy(schemas),
                )
            )
            if len(out) >= target_problems:
                return out
    return out


def sample_balanced_problems(
    pools: Dict[int, List[Problem]], total: int, seed: int
) -> List[Problem]:
    depths = sorted(pools.keys())
    if not depths:
        raise RuntimeError("No depth pools available for sampling.")

    rng = random.Random(seed)
    k = len(depths)
    base = total // k
    rem = total % k
    target_per_depth = {d: base for d in depths}
    for d in depths[:rem]:
        target_per_depth[d] += 1

    sampled: List[Problem] = []
    for depth in depths:
        pool = pools[depth]
        need = target_per_depth[depth]
        if len(pool) < need:
            raise RuntimeError(
                f"Depth {depth} has only {len(pool)} parsed problems, need {need}."
            )
        picked = rng.sample(pool, need)
        sampled.extend(copy.deepcopy(picked))
    rng.shuffle(sampled)
    return sampled


def problem_exposure_profile(
    problem: Problem,
    max_steps: int,
    breadth_limit: int,
) -> Dict[str, object]:
    # Reachability analysis on parsed schemas, tracking whether a derivation path
    # contains at least one TRANSITIVE-family rule.
    without_depth: Dict[str, int] = {}
    with_depth: Dict[str, int] = {}

    for fact in problem.facts:
        prop = normalize_prop(fact)
        if prop:
            if prop not in without_depth:
                without_depth[prop] = 0

    active_rules = list(problem.schemas)
    if len(active_rules) > breadth_limit:
        active_rules = active_rules[:breadth_limit]

    for _step in range(1, max_steps + 1):
        any_new = False
        working_props = set(without_depth.keys()) | set(with_depth.keys())
        entity_pool = {
            ent for prop in working_props for ent in prop_entities(prop) if ent
        }
        entities = sorted(entity_pool)

        for schema in active_rules:
            if not schema.antecedents:
                continue

            instantiations: List[Tuple[Tuple[str, ...], str]] = []
            has_var = any(prop_has_variable(a) for a in schema.antecedents) or prop_has_variable(
                schema.consequent
            )
            if has_var:
                var_ants = [a for a in schema.antecedents if prop_has_variable(a)]
                if var_ants:
                    candidate_sets = [
                        set(candidate_bindings_for_pattern(ant, working_props)) for ant in var_ants
                    ]
                    candidate_bindings = set.intersection(*candidate_sets) if candidate_sets else set()
                else:
                    candidate_bindings = set(entities)
                if not candidate_bindings and entities:
                    candidate_bindings = set(entities)
                if not candidate_bindings:
                    continue

                for binding in sorted(candidate_bindings):
                    ants = tuple(substitute_variable(a, binding) for a in schema.antecedents)
                    if not all(ant in working_props for ant in ants):
                        continue
                    cons = substitute_variable(schema.consequent, binding)
                    instantiations.append((ants, cons))
            else:
                if all(ant in working_props for ant in schema.antecedents):
                    instantiations.append((schema.antecedents, schema.consequent))

            for ant_tuple, consequent_prop in instantiations:
                ant_any_depths: List[int] = []
                for ant in ant_tuple:
                    d_no = without_depth.get(ant)
                    d_with = with_depth.get(ant)
                    cands = [d for d in (d_no, d_with) if d is not None]
                    if not cands:
                        ant_any_depths = []
                        break
                    ant_any_depths.append(min(cands))
                if not ant_any_depths:
                    continue

                cand_without: Optional[int] = None
                cand_with: Optional[int] = None
                if schema.family == "TRANSITIVE":
                    cand_with = max(ant_any_depths) + 1
                else:
                    ant_without_depths = [without_depth.get(ant) for ant in ant_tuple]
                    if all(d is not None for d in ant_without_depths):
                        cand_without = max(int(d) for d in ant_without_depths) + 1

                    transitive_mix_depths: List[int] = []
                    for i, ant in enumerate(ant_tuple):
                        d_with = with_depth.get(ant)
                        if d_with is None:
                            continue
                        others: List[int] = []
                        valid = True
                        for j, other_ant in enumerate(ant_tuple):
                            if i == j:
                                continue
                            d_no = without_depth.get(other_ant)
                            d_oth_with = with_depth.get(other_ant)
                            cands = [d for d in (d_no, d_oth_with) if d is not None]
                            if not cands:
                                valid = False
                                break
                            others.append(min(cands))
                        if not valid:
                            continue
                        cand = max([d_with, *others]) + 1
                        transitive_mix_depths.append(cand)
                    if transitive_mix_depths:
                        cand_with = min(transitive_mix_depths)

                if cand_without is not None:
                    old = without_depth.get(consequent_prop)
                    if old is None or cand_without < old:
                        without_depth[consequent_prop] = cand_without
                        any_new = True
                if cand_with is not None:
                    old = with_depth.get(consequent_prop)
                    if old is None or cand_with < old:
                        with_depth[consequent_prop] = cand_with
                        any_new = True

        if not any_new:
            break

    q = normalize_prop(problem.question)
    q_neg = negate_prop(q)
    answer_prop = q if int(problem.answer) == 1 else q_neg
    answer_neg_prop = q_neg if int(problem.answer) == 1 else q
    working_props = set(without_depth.keys()) | set(with_depth.keys())
    return {
        "answer_reachable": answer_prop in working_props,
        "answer_transitive_reachable": answer_prop in with_depth,
        "counter_reachable": answer_neg_prop in working_props,
        "question_reachable": q in working_props,
        "question_neg_reachable": q_neg in working_props,
        "question_transitive_reachable": q in with_depth,
        "question_neg_transitive_reachable": q_neg in with_depth,
        "answer_depth": min(
            [d for d in (without_depth.get(answer_prop), with_depth.get(answer_prop)) if d is not None],
            default=None,
        ),
        "answer_transitive_depth": with_depth.get(answer_prop),
    }


def apply_problem_selection_filter(
    pools: Dict[int, List[Problem]],
    cfg: Dict[str, object],
    dataset_key: str,
) -> Dict[str, object]:
    rt_cfg = cfg[dataset_key]
    selection_cfg = rt_cfg.get("selection", {})
    if not isinstance(selection_cfg, dict):
        selection_cfg = {}

    mode = str(selection_cfg.get("mode", "all")).strip().lower()
    if mode in ("all", "none", "off", "disabled"):
        return {"enabled": False, "mode": "all", "per_depth": {}}

    reason_cfg = cfg.get("reasoner", {})
    if not isinstance(reason_cfg, dict):
        reason_cfg = {}
    max_steps = int(reason_cfg.get("max_steps", 7))
    breadth_limit = int(reason_cfg.get("breadth_limit", 50))
    min_answer_depth = int(selection_cfg.get("min_answer_depth", 0))
    min_answer_transitive_depth = int(selection_cfg.get("min_answer_transitive_depth", 0))

    mode_to_key = {
        "answer_reachable": "answer_reachable",
        "answer_transitive_reachable": "answer_transitive_reachable",
        "question_side_transitive_reachable": "question_transitive_reachable",
    }
    if mode not in mode_to_key:
        raise ValueError(
            f"Unsupported {dataset_key}.selection.mode `{mode}`. "
            "Use one of: all, answer_reachable, answer_transitive_reachable, question_side_transitive_reachable."
        )
    key = mode_to_key[mode]

    per_depth: Dict[str, object] = {}
    for depth, problems in pools.items():
        keep: List[Problem] = []
        stats = {
            "before": len(problems),
            "answer_reachable": 0,
            "answer_transitive_reachable": 0,
            "answer_depth_ge_min": 0,
            "answer_transitive_depth_ge_min": 0,
            "question_reachable": 0,
            "question_transitive_reachable": 0,
            "question_neg_reachable": 0,
            "question_neg_transitive_reachable": 0,
        }
        for problem in problems:
            profile = problem_exposure_profile(
                problem=problem,
                max_steps=max_steps,
                breadth_limit=breadth_limit,
            )
            for k in (
                "answer_reachable",
                "answer_transitive_reachable",
                "question_reachable",
                "question_transitive_reachable",
                "question_neg_reachable",
                "question_neg_transitive_reachable",
            ):
                if profile.get(k, False):
                    stats[k] += 1
            answer_depth = profile.get("answer_depth")
            answer_transitive_depth = profile.get("answer_transitive_depth")
            depth_ok = True
            if min_answer_depth > 0:
                depth_ok = (
                    isinstance(answer_depth, int)
                    and answer_depth >= min_answer_depth
                )
                if depth_ok:
                    stats["answer_depth_ge_min"] += 1
            if min_answer_transitive_depth > 0:
                trans_depth_ok = (
                    isinstance(answer_transitive_depth, int)
                    and answer_transitive_depth >= min_answer_transitive_depth
                )
                if trans_depth_ok:
                    stats["answer_transitive_depth_ge_min"] += 1
                depth_ok = depth_ok and trans_depth_ok

            if profile.get(key, False) and depth_ok:
                keep.append(problem)
        pools[depth] = keep
        stats["after"] = len(keep)
        stats["kept_rate"] = len(keep) / max(1, len(problems))
        per_depth[str(depth)] = stats

    return {"enabled": True, "mode": mode, "per_depth": per_depth}


def load_problems_base(cfg: Dict[str, object], dataset_key: str) -> List[Problem]:
    if dataset_key not in cfg:
        raise KeyError(f"Missing dataset config block: `{dataset_key}`")
    rt_cfg = cfg[dataset_key]
    depths = [int(d) for d in rt_cfg["depths"]]
    target_total = int(rt_cfg["num_problems"])
    max_rules_per_problem = int(rt_cfg["max_rules_per_problem"])
    sample_seed = int(rt_cfg["sample_seed"])
    local_jsonl_dir = rt_cfg.get("local_jsonl_dir")
    depth_aliases = rt_cfg.get("depth_aliases", {})
    parse_buffer_factor = int(rt_cfg.get("parse_buffer_factor", 3))
    max_rows_scan_per_depth = int(rt_cfg.get("max_rows_scan_per_depth", 120000))

    sorted_depths = sorted(depths)
    k = len(sorted_depths)
    base_need = target_total // max(1, k)
    rem = target_total % max(1, k)
    need_by_depth = {d: base_need for d in sorted_depths}
    for d in sorted_depths[:rem]:
        need_by_depth[d] += 1

    pools: Dict[int, List[Problem]] = {}
    if local_jsonl_dir:
        base = Path(str(local_jsonl_dir))
        if not base.exists():
            raise FileNotFoundError(f"Configured local_jsonl_dir does not exist: {base}")

        for depth in depths:
            files = sorted(base.glob(f"*d{depth}*.jsonl")) + sorted(
                base.glob(f"*depth{depth}*.jsonl")
            )
            if not files:
                raise FileNotFoundError(
                    f"No local JSONL file found for depth {depth} under {base}"
                )

            rows: List[Dict[str, object]] = []
            for fp in files:
                with open(fp, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        rows.append(json.loads(line))
            prefix = f"d{depth}_local"
            pools[depth] = flatten_ruletaker_rows_limited(
                rows=rows,
                depth=depth,
                max_rules_per_problem=max_rules_per_problem,
                key_prefix=prefix,
                target_problems=max(
                    need_by_depth.get(depth, base_need) * parse_buffer_factor,
                    need_by_depth.get(depth, base_need),
                ),
                max_rows_scan=max_rows_scan_per_depth,
            )
    else:
        if not DATASETS_AVAILABLE:
            raise RuntimeError(
                "`datasets` package is unavailable and no local_jsonl_dir was provided."
            )
        dataset_name = str(rt_cfg["dataset_name"])
        split_pref = [str(s) for s in rt_cfg["split_preference"]]
        configs_by_depth = select_depth_configs(dataset_name, depths)

        for depth in depths:
            config_name = configs_by_depth.get(depth)
            dset = load_any_split(dataset_name, config_name, split_pref)
            prefix = f"d{depth}_{config_name or 'default'}"
            depth_filter_token = str(depth_aliases.get(str(depth), f"depth-{depth}"))

            if isinstance(dset, Dataset) and "config" in dset.column_names:
                cfg_col = dset["config"]
                idx = [
                    i
                    for i, cfgv in enumerate(cfg_col)
                    if normalize_text(str(cfgv)) == normalize_text(depth_filter_token)
                ]
                if idx:
                    dset = dset.select(idx)
                    depth_filter_token = None

            pools[depth] = flatten_ruletaker_rows_limited(
                rows=dset,
                depth=depth,
                max_rules_per_problem=max_rules_per_problem,
                key_prefix=prefix,
                target_problems=max(
                    need_by_depth.get(depth, base_need) * parse_buffer_factor,
                    need_by_depth.get(depth, base_need),
                ),
                max_rows_scan=max_rows_scan_per_depth,
                depth_filter_token=depth_filter_token,
            )

    selection_stats = apply_problem_selection_filter(pools, cfg, dataset_key=dataset_key)
    if selection_stats.get("enabled"):
        for depth in sorted(pools.keys()):
            d = selection_stats["per_depth"].get(str(depth), {})
            print(
                "[selection]",
                f"dataset={dataset_key}",
                f"mode={selection_stats.get('mode')}",
                f"depth={depth}",
                f"kept={d.get('after', 0)}/{d.get('before', 0)}",
                f"answer_transitive={d.get('answer_transitive_reachable', 0)}",
            )

    return sample_balanced_problems(pools, target_total, sample_seed)


# -----------------------------
# Noise injection
# -----------------------------


def corrupt_schema(schema: Schema, replacement_consequent: Optional[str] = None) -> None:
    schema.corrupted = True
    if replacement_consequent:
        schema.consequent = normalize_prop(replacement_consequent)
    else:
        schema.consequent = negate_prop(schema.consequent)

    # Keep text shape recognizable while flipping meaning.
    parsed = parse_rule_sentence(schema.text)
    if parsed is not None:
        ants, _, _ = parsed
        ant_text = " and ".join(ants)
        schema.text = f"if {ant_text} then {schema.consequent}"
    else:
        schema.text = f"if {' and '.join(schema.antecedents)} then {schema.consequent}"


def corrupt_schema_redirect(schema: Schema, redirect_target: str) -> None:
    """Redirect-based corruption used by the synthetic mechanism check.

    Replaces the consequent with a downstream-consumable target proposition,
    so the corrupted output can propagate through later schemas.
    """
    schema.corrupted = True
    schema.consequent = normalize_prop(redirect_target)
    schema.text = f"if {' and '.join(schema.antecedents)} then {schema.consequent}"


def apply_noise(
    base_problems: List[Problem],
    condition: str,
    seed: int,
    cfg: Dict[str, object],
) -> List[Problem]:
    noise_cfg = cfg["noise"]
    random_rate = float(noise_cfg["random_rate"])
    clustered_rate = float(noise_cfg["clustered_rate"])
    clustered_family = str(noise_cfg["clustered_family"])
    corruption_mode = "redirect" if "redirect" in condition else "negate"
    if condition == "clean":
        base_condition = "clean"
    elif condition.startswith("random"):
        base_condition = "random"
    elif condition.startswith("clustered"):
        base_condition = "clustered"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    rng = random.Random(deterministic_seed("noise", condition, seed))
    problems = copy.deepcopy(base_problems)

    for problem in problems:
        fact_props = {normalize_prop(f) for f in problem.facts if normalize_prop(f)}
        antecedent_counts: Dict[str, int] = {}
        for cand_schema in problem.schemas:
            for ant in cand_schema.antecedents:
                ant_n = normalize_prop(ant)
                if ant_n:
                    antecedent_counts[ant_n] = antecedent_counts.get(ant_n, 0) + 1

        for idx, schema in enumerate(problem.schemas):
            schema.corrupted = False
            should_corrupt = False
            if base_condition == "clean":
                should_corrupt = False
            elif base_condition == "random":
                should_corrupt = rng.random() < random_rate
            elif base_condition == "clustered":
                if schema.family == clustered_family:
                    should_corrupt = rng.random() < clustered_rate
            else:
                raise ValueError(f"Unknown base condition: {base_condition}")

            if should_corrupt:
                replacement: Optional[str] = None
                if corruption_mode == "redirect":
                    original_consequent = normalize_prop(schema.consequent)
                    redirect_candidates: List[Tuple[int, str]] = []
                    for j, cand_schema in enumerate(problem.schemas):
                        if j == idx:
                            continue
                        for ant in cand_schema.antecedents:
                            ant_n = normalize_prop(ant)
                            if not ant_n:
                                continue
                            if ant_n == original_consequent:
                                continue
                            if ant_n == negate_prop(original_consequent):
                                continue
                            if ant_n.startswith("not|"):
                                continue
                            score = antecedent_counts.get(ant_n, 1)
                            if ant_n not in fact_props:
                                score += 100
                            redirect_candidates.append((score, ant_n))
                    if redirect_candidates:
                        redirect_candidates.sort(key=lambda x: (-x[0], x[1]))
                        top_k = min(5, len(redirect_candidates))
                        choice_idx = rng.randrange(top_k)
                        replacement = redirect_candidates[choice_idx][1]

                if replacement is not None and corruption_mode == "redirect":
                    corrupt_schema_redirect(schema, replacement)
                else:
                    corrupt_schema(schema)

    return problems


# -----------------------------
# Reliability estimator
# -----------------------------


def build_schema_records(problems: List[Problem]) -> List[SchemaRecord]:
    records: List[SchemaRecord] = []
    for problem in problems:
        for schema in problem.schemas:
            text = f"{schema.text} [SEP] family={schema.family} [SEP] source={schema.source_id}"
            label = 0 if schema.corrupted else 1
            records.append(
                SchemaRecord(
                    schema_id=schema.schema_id,
                    text=text,
                    family=schema.family,
                    source_id=schema.source_id,
                    label=label,
                )
            )
    return records


def split_records(
    records: List[SchemaRecord], seed: int
) -> Tuple[List[SchemaRecord], List[SchemaRecord], List[SchemaRecord]]:
    rng = random.Random(deterministic_seed("split", seed))
    records = records[:]
    rng.shuffle(records)
    n = len(records)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]
    return train, val, test


def inject_label_noise(records: List[SchemaRecord], rate: float, seed: int) -> List[SchemaRecord]:
    if rate <= 0:
        return copy.deepcopy(records)
    rng = random.Random(deterministic_seed("label_noise", seed, rate))
    out = copy.deepcopy(records)
    for rec in out:
        if rng.random() < rate:
            rec.label = 1 - rec.label
    return out


def train_count_estimator(
    records: List[SchemaRecord],
    threshold: float,
    seed: int,
    label_noise_rate: float,
) -> EstimatorModel:
    train, val, test = split_records(records, seed)
    train = inject_label_noise(train, label_noise_rate, seed)

    source_sum: Dict[str, float] = {}
    source_count: Dict[str, int] = {}
    family_sum: Dict[str, float] = {}
    family_count: Dict[str, int] = {}

    for rec in train:
        source_sum[rec.source_id] = source_sum.get(rec.source_id, 0.0) + rec.label
        source_count[rec.source_id] = source_count.get(rec.source_id, 0) + 1
        family_sum[rec.family] = family_sum.get(rec.family, 0.0) + rec.label
        family_count[rec.family] = family_count.get(rec.family, 0) + 1

    source_scores = {
        src: source_sum[src] / max(1, source_count[src]) for src in source_sum.keys()
    }
    family_scores = {
        fam: family_sum[fam] / max(1, family_count[fam]) for fam in family_sum.keys()
    }
    global_score = statistics.mean([rec.label for rec in train]) if train else 0.5

    def predict(rec: SchemaRecord) -> int:
        score = source_scores.get(rec.source_id, family_scores.get(rec.family, global_score))
        return 1 if score >= threshold else 0

    val_acc = (
        sum(1 for rec in val if predict(rec) == rec.label) / max(1, len(val)) if val else 0.0
    )
    test_acc = (
        sum(1 for rec in test if predict(rec) == rec.label) / max(1, len(test)) if test else 0.0
    )

    return EstimatorModel(
        threshold=threshold,
        backend="count_fallback",
        source_scores=source_scores,
        family_scores=family_scores,
        global_score=global_score,
        metrics={"val_acc": val_acc, "test_acc": test_acc},
    )


def train_reliability_estimator(
    records: List[SchemaRecord],
    cfg: Dict[str, object],
    seed: int,
    label_noise_rate: float,
) -> EstimatorModel:
    est_cfg = cfg["estimator"]
    backend = str(est_cfg.get("backend", "count_fallback"))
    threshold = float(est_cfg["threshold"])

    if backend != "count_fallback":
        raise ValueError(
            f"Unsupported estimator backend for paper pipeline: {backend}. "
            "Expected: count_fallback."
        )

    return train_count_estimator(
        records=records,
        threshold=threshold,
        seed=seed,
        label_noise_rate=label_noise_rate,
    )


# -----------------------------
# Reasoning engine
# -----------------------------


def clarity_value(family: Optional[str]) -> float:
    if family == "IMPLICATION":
        return 1.0
    if family == "TRANSITIVE":
        return 0.7
    if family == "DEFAULT":
        return 0.4
    return 1.0


def node_rank(node: DerivationNode) -> Tuple[float, float, float]:
    return (node.clarity, node.reliability, node.confidence)


def better(new_node: DerivationNode, old_node: DerivationNode) -> bool:
    return node_rank(new_node) > node_rank(old_node)


def infer_problem(
    problem: Problem,
    estimator: EstimatorModel,
    model: str,
    cfg: Dict[str, object],
    seed: int,
) -> InferenceResult:
    reason_cfg = cfg["reasoner"]
    max_steps = int(reason_cfg["max_steps"])
    breadth_limit = int(reason_cfg["breadth_limit"])
    decision_conf_thr = float(reason_cfg.get("decision_confidence_threshold", 0.5))
    default_policy = str(reason_cfg.get("default_policy", "closed_world"))
    tau = float(estimator.threshold)

    rng = random.Random(deterministic_seed("infer", seed, model, problem.problem_id))

    nodes: Dict[str, DerivationNode] = {}
    best_committed: Dict[str, str] = {}
    node_counter = 0

    def add_node(node: DerivationNode) -> str:
        nonlocal node_counter
        node_counter += 1
        node_id = f"N{node_counter:04d}"
        node.node_id = node_id
        nodes[node_id] = node
        return node_id

    # Initialize facts.
    for fact in problem.facts:
        prop = normalize_prop(fact)
        if not prop:
            continue
        n = DerivationNode(
            node_id="",
            proposition=prop,
            parent_ids=tuple(),
            rule_id=None,
            family=None,
            source_id=None,
            reliability=1.0,
            clarity=1.0,
            confidence=1.0,
            corrupted=False,
            step=0,
            committed=True,
        )
        nid = add_node(n)
        if prop not in best_committed or better(n, nodes[best_committed[prop]]):
            best_committed[prop] = nid

    # Prepare active rules.
    active_rules = copy.deepcopy(problem.schemas)
    rng.shuffle(active_rules)
    removed_schema_ids: List[str] = []
    transient_blocked_total = 0
    consumed_transient_node_ids: set[str] = set()

    if model == "schema_gating":
        gated = []
        for schema in active_rules:
            score = estimator.score(schema)
            if score < tau:
                removed_schema_ids.append(schema.schema_id)
            else:
                gated.append(schema)
        active_rules = gated

    # Breadth cutoff on rule pool (identical cap logic across models).
    if len(active_rules) > breadth_limit:
        active_rules = active_rules[:breadth_limit]

    window1_active_transient: Dict[str, str] = {}

    for step in range(1, max_steps + 1):
        any_new = False
        working_props = set(best_committed.keys())
        if model == "aspic_v2_persistent":
            working_props.update(window1_active_transient.keys())
        pending_best: Dict[str, str] = {}
        window1_next_transient: Dict[str, str] = {}
        transient_props_this_step: set[str] = set()

        entity_pool = {
            ent for prop in working_props for ent in prop_entities(prop) if ent
        }
        entities = sorted(entity_pool)

        for schema in active_rules:
            if not schema.antecedents:
                continue

            instantiations: List[Tuple[Tuple[str, ...], str, str]] = []
            has_var = any(prop_has_variable(a) for a in schema.antecedents) or prop_has_variable(
                schema.consequent
            )
            if has_var:
                var_ants = [a for a in schema.antecedents if prop_has_variable(a)]
                if var_ants:
                    candidate_sets = [
                        set(candidate_bindings_for_pattern(ant, working_props)) for ant in var_ants
                    ]
                    candidate_bindings = set.intersection(*candidate_sets) if candidate_sets else set()
                else:
                    candidate_bindings = set(entities)

                if not candidate_bindings and entities:
                    candidate_bindings = set(entities)
                if not candidate_bindings:
                    continue

                for binding in sorted(candidate_bindings):
                    ants = tuple(substitute_variable(a, binding) for a in schema.antecedents)
                    if not all(ant in working_props for ant in ants):
                        continue
                    cons = substitute_variable(schema.consequent, binding)
                    instantiations.append((ants, cons, binding))
            else:
                if all(ant in working_props for ant in schema.antecedents):
                    instantiations.append((schema.antecedents, schema.consequent, ""))

            for ant_tuple, consequent_prop, binding in instantiations:
                parent_ids: List[str] = []
                parent_conf = 1.0
                consumed_transient = False
                consumed_transient_source_ids: set[str] = set()
                for ant in ant_tuple:
                    pending_pid = pending_best.get(ant)
                    window1_pid = window1_active_transient.get(ant)
                    pid = pending_pid or window1_pid or best_committed.get(ant)
                    if pid is None:
                        parent_ids = []
                        break
                    if pending_pid is not None and ant in transient_props_this_step:
                        consumed_transient = True
                        consumed_transient_source_ids.add(pending_pid)
                    if (
                        model == "aspic_v2_persistent"
                        and pending_pid is None
                        and window1_pid is not None
                    ):
                        consumed_transient = True
                        consumed_transient_source_ids.add(window1_pid)
                    parent_ids.append(pid)
                    parent_conf = min(parent_conf, nodes[pid].confidence)
                if not parent_ids:
                    continue

                rel = estimator.score(schema)
                conf = parent_conf * rel
                rule_instance_id = schema.schema_id if not binding else f"{schema.schema_id}@{binding}"
                node = DerivationNode(
                    node_id="",
                    proposition=consequent_prop,
                    parent_ids=tuple(parent_ids),
                    rule_id=rule_instance_id,
                    family=schema.family,
                    source_id=schema.source_id,
                    reliability=rel,
                    clarity=clarity_value(schema.family),
                    confidence=conf,
                    corrupted=schema.corrupted,
                    step=step,
                    committed=True,
                )
                nid = add_node(node)

                blocked = model in ("aspic_v2", "aspic_v2_persistent") and rel < tau
                if blocked:
                    # Approximate chain-intercepting ASPIC+ behavior:
                    # the undercutter fires after schema selection/matching. We represent this
                    # by creating a transient node excluded from the committed knowledge set.
                    node.committed = False
                    transient_blocked_total += 1
                    if model == "aspic_v2":
                        # Same-step transient visibility.
                        existing_pid = pending_best.get(consequent_prop)
                        if existing_pid is None or better(node, nodes[existing_pid]):
                            pending_best[consequent_prop] = nid
                            transient_props_this_step.add(consequent_prop)
                        working_props.add(consequent_prop)
                    else:
                        # Window-1 variant: visible in the next step only, then retracted.
                        existing_pid = window1_next_transient.get(consequent_prop)
                        if existing_pid is None or better(node, nodes[existing_pid]):
                            window1_next_transient[consequent_prop] = nid
                    continue

                if consumed_transient:
                    consumed_transient_node_ids.update(consumed_transient_source_ids)

                existing_cid = best_committed.get(consequent_prop)
                if existing_cid is None or better(node, nodes[existing_cid]):
                    best_committed[consequent_prop] = nid
                    working_props.add(consequent_prop)
                    any_new = True

        if model == "aspic_v2_persistent":
            window1_active_transient = window1_next_transient

        if not any_new and not (model == "aspic_v2_persistent" and window1_active_transient):
            break

    q = normalize_prop(problem.question)
    q_neg = negate_prop(q)

    def rank_or_none(nid: Optional[str]) -> Tuple[float, float, float]:
        if nid is None:
            return (-1.0, -1.0, -1.0)
        return node_rank(nodes[nid])

    def has_corrupted_intermediate_local(root_node_id: Optional[str]) -> bool:
        if root_node_id is None:
            return False
        root = nodes.get(root_node_id)
        if root is None:
            return False
        stack = list(root.parent_ids)  # exclude final node by definition
        seen: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            node = nodes.get(nid)
            if node is None:
                continue
            if node.rule_id is not None and node.corrupted:
                return True
            for pid in node.parent_ids:
                if pid not in seen:
                    stack.append(pid)
        return False

    def visible_best(prop: str, blocked_ids: set[str]) -> Optional[str]:
        nid = best_committed.get(prop)
        if nid in blocked_ids:
            return None
        return nid

    def decide(blocked_ids: set[str]) -> Tuple[int, Optional[str], bool]:
        pos_id = visible_best(q, blocked_ids)
        neg_id = visible_best(q_neg, blocked_ids)
        pos_rank = rank_or_none(pos_id)
        neg_rank = rank_or_none(neg_id)

        if pos_rank > neg_rank and pos_rank[2] >= decision_conf_thr:
            return 1, pos_id, False
        if neg_rank >= pos_rank and neg_rank[2] >= decision_conf_thr:
            return 0, neg_id, False

        # If neither side has enough support, fall back to configured default policy.
        if default_policy == "closed_world":
            # Closed-world fallback used by RuleTaker-style data:
            # if a positive atom is not derivable, assume its negation.
            if q.startswith("not|"):
                pos_atom = q[4:]
                pos_atom_id = visible_best(pos_atom, blocked_ids)
                if pos_atom_id is not None:
                    return 0, pos_atom_id, True
                return 1, neg_id, True
            if pos_id is not None:
                return 1, pos_id, True
            return 0, neg_id, True

        if default_policy == "random_tie":
            tie_rng = random.Random(deterministic_seed("default_pred", problem.problem_id))
            pred = tie_rng.randint(0, 1)
            pred_node = pos_id if pred == 1 else neg_id
            return pred, pred_node, True

        raise ValueError(f"Unsupported default_policy: {default_policy}")

    used_default_decision = False
    blocked_node_ids: set[str] = set()
    while True:
        pred, pred_node, used_default = decide(blocked_node_ids)
        used_default_decision = used_default_decision or used_default
        if model != "aspic_v1_terminal":
            break
        if pred_node is None:
            break
        if not has_corrupted_intermediate_local(pred_node):
            break
        blocked_node_ids.add(pred_node)
        # Prevent pathological cycles if both sides repeatedly fail and no
        # alternative support exists.
        if len(blocked_node_ids) > 8:
            break

    return InferenceResult(
        predicted_label=pred,
        predicted_node_id=pred_node,
        nodes=nodes,
        best_committed=best_committed,
        removed_schema_ids=removed_schema_ids,
        used_default_decision=used_default_decision,
        transient_blocked_total=transient_blocked_total,
        transient_consumed_total=len(consumed_transient_node_ids),
    )


def has_corrupted_intermediate(
    result: InferenceResult,
    root_node_id: Optional[str],
    has_any_corruption: bool,
) -> bool:
    if root_node_id is None or not has_any_corruption:
        return False

    root = result.nodes.get(root_node_id)
    if root is None:
        return False

    stack = list(root.parent_ids)  # exclude final node by definition
    seen = set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = result.nodes.get(nid)
        if node is None:
            continue
        if node.rule_id is not None and node.corrupted:
            return True
        for pid in node.parent_ids:
            if pid not in seen:
                stack.append(pid)
    return False


def derivation_chain_length(result: InferenceResult, node_id: Optional[str]) -> int:
    """Number of rule-application steps along the strongest ancestor path."""
    if node_id is None:
        return 0
    memo: Dict[str, int] = {}

    def walk(nid: str) -> int:
        cached = memo.get(nid)
        if cached is not None:
            return cached
        node = result.nodes.get(nid)
        if node is None:
            memo[nid] = 0
            return 0
        parent_depth = 0
        for pid in node.parent_ids:
            parent_depth = max(parent_depth, walk(pid))
        depth = parent_depth + 1 if node.rule_id is not None else parent_depth
        memo[nid] = depth
        return depth

    return walk(node_id)


# -----------------------------
# Metrics and experiment loops
# -----------------------------


def source_reliability_map(problems: List[Problem]) -> Dict[str, float]:
    # Reliability defined by clean ratio (>= 0.85 => reliable).
    counts: Dict[str, int] = {}
    clean: Dict[str, int] = {}
    for p in problems:
        for s in p.schemas:
            counts[s.source_id] = counts.get(s.source_id, 0) + 1
            if not s.corrupted:
                clean[s.source_id] = clean.get(s.source_id, 0) + 1
    return {src: clean.get(src, 0) / max(1, cnt) for src, cnt in counts.items()}


def evaluate_model_on_condition(
    problems: List[Problem],
    estimator: EstimatorModel,
    model: str,
    cfg: Dict[str, object],
    seed: int,
    reliable_sources: Dict[str, float],
) -> Dict[str, object]:
    total = len(problems)
    correct = 0
    incorrect = 0
    incorrect_with_chain = 0
    corrupted_chain = 0
    errors_cwa = 0
    errors_chain_linked = 0
    errors_direct = 0

    derivable_total = 0
    derivable_correct = 0
    derivable_incorrect = 0
    derivable_errors_chain_linked = 0
    derivable_errors_direct = 0

    removed_total = 0
    schema_total = 0
    reliable_total = 0
    reliable_removed = 0
    default_decision_total = 0
    transient_blocked_total = 0
    transient_consumed_total = 0

    for p in problems:
        res = infer_problem(
            problem=p,
            estimator=estimator,
            model=model,
            cfg=cfg,
            seed=seed,
        )
        pred = res.predicted_label
        is_derivable = (res.predicted_node_id is not None) and (not res.used_default_decision)
        if is_derivable:
            derivable_total += 1

        if res.used_default_decision:
            default_decision_total += 1
        transient_blocked_total += res.transient_blocked_total
        transient_consumed_total += res.transient_consumed_total
        if pred == p.answer:
            correct += 1
            if is_derivable:
                derivable_correct += 1
        else:
            incorrect += 1
            if is_derivable:
                derivable_incorrect += 1

            if res.used_default_decision:
                errors_cwa += 1

            if res.predicted_node_id is not None:
                incorrect_with_chain += 1
                chain_depth = derivation_chain_length(res, res.predicted_node_id)
                if not res.used_default_decision:
                    if chain_depth >= 2:
                        errors_chain_linked += 1
                        if is_derivable:
                            derivable_errors_chain_linked += 1
                    else:
                        errors_direct += 1
                        if is_derivable:
                            derivable_errors_direct += 1
                has_corrupt = any(s.corrupted for s in p.schemas)
                if has_corrupted_intermediate(res, res.predicted_node_id, has_corrupt):
                    corrupted_chain += 1

        if model == "schema_gating":
            removed_set = set(res.removed_schema_ids)
            removed_total += len(removed_set)
            schema_total += len(p.schemas)
            for s in p.schemas:
                if reliable_sources.get(s.source_id, 1.0) >= 0.85:
                    reliable_total += 1
                    if s.schema_id in removed_set:
                        reliable_removed += 1

    acc = correct / max(1, total)
    chain_error_rate = incorrect_with_chain / max(1, incorrect)
    corrupted_cascade_rate = corrupted_chain / max(1, incorrect)
    error_cwa_rate = errors_cwa / max(1, incorrect)
    error_chain_linked_rate = errors_chain_linked / max(1, incorrect)
    error_direct_rate = errors_direct / max(1, incorrect)
    transient_consumption_rate = transient_consumed_total / max(1, transient_blocked_total)
    derivable_accuracy = derivable_correct / max(1, derivable_total)
    derivable_chain_linked_error_rate = derivable_errors_chain_linked / max(1, derivable_incorrect)
    derivable_direct_error_rate = derivable_errors_direct / max(1, derivable_incorrect)

    all_metrics: Dict[str, float] = {
        "accuracy": acc,
        # Paper-definition primary metric.
        "cascade_rate": corrupted_cascade_rate,
        "chain_error_rate": chain_error_rate,
        "corrupted_cascade_rate": corrupted_cascade_rate,
        "errors_cwa": float(errors_cwa),
        "errors_chain_linked": float(errors_chain_linked),
        "errors_direct": float(errors_direct),
        "error_cwa_rate": error_cwa_rate,
        "error_chain_linked_rate": error_chain_linked_rate,
        "error_direct_rate": error_direct_rate,
        "incorrect": float(incorrect),
        "incorrect_with_chain": float(incorrect_with_chain),
        "corrupted_chain": float(corrupted_chain),
        "transient_blocked_total": float(transient_blocked_total),
        "transient_consumed_total": float(transient_consumed_total),
        "transient_consumption_rate": transient_consumption_rate,
        "schema_total": float(schema_total),
        "removed_total": float(removed_total),
        "reliable_total": float(reliable_total),
        "reliable_removed": float(reliable_removed),
        "default_decision_total": float(default_decision_total),
        "derivable_total": float(derivable_total),
        "derivable_correct": float(derivable_correct),
        "derivable_incorrect": float(derivable_incorrect),
        "derivable_accuracy": derivable_accuracy,
        "derivable_chain_linked_error_rate": derivable_chain_linked_error_rate,
        "derivable_direct_error_rate": derivable_direct_error_rate,
    }
    derivable_metrics: Dict[str, float] = {
        "total": float(derivable_total),
        "correct": float(derivable_correct),
        "incorrect": float(derivable_incorrect),
        "accuracy": derivable_accuracy,
        "chain_linked_error_rate": derivable_chain_linked_error_rate,
        "direct_error_rate": derivable_direct_error_rate,
    }
    all_payload: Dict[str, float] = dict(all_metrics)
    all_payload.update({"all": all_metrics, "derivable_only": derivable_metrics})
    return all_payload


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def aggregate_seed_rows(rows: List[Dict[str, object]]) -> Dict[str, float]:
    acc_m, acc_s = mean_std([r["accuracy"] for r in rows])
    cas_m, cas_s = mean_std([r["cascade_rate"] for r in rows])
    chain_m, chain_s = mean_std([r.get("chain_error_rate", 0.0) for r in rows])
    trans_m, trans_s = mean_std([r.get("transient_consumption_rate", 0.0) for r in rows])
    cwa_m, cwa_s = mean_std([r.get("error_cwa_rate", 0.0) for r in rows])
    chain_linked_m, chain_linked_s = mean_std(
        [r.get("error_chain_linked_rate", 0.0) for r in rows]
    )
    direct_m, direct_s = mean_std([r.get("error_direct_rate", 0.0) for r in rows])
    deriv_acc_m, deriv_acc_s = mean_std([r.get("derivable_accuracy", 0.0) for r in rows])
    deriv_chain_m, deriv_chain_s = mean_std(
        [r.get("derivable_chain_linked_error_rate", 0.0) for r in rows]
    )
    deriv_direct_m, deriv_direct_s = mean_std(
        [r.get("derivable_direct_error_rate", 0.0) for r in rows]
    )
    return {
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "cascade_mean": cas_m,
        "cascade_std": cas_s,
        "chain_error_mean": chain_m,
        "chain_error_std": chain_s,
        "error_cwa_mean": cwa_m,
        "error_cwa_std": cwa_s,
        "error_chain_linked_mean": chain_linked_m,
        "error_chain_linked_std": chain_linked_s,
        "error_direct_mean": direct_m,
        "error_direct_std": direct_s,
        "derivable_accuracy_mean": deriv_acc_m,
        "derivable_accuracy_std": deriv_acc_s,
        "derivable_chain_linked_mean": deriv_chain_m,
        "derivable_chain_linked_std": deriv_chain_s,
        "derivable_direct_mean": deriv_direct_m,
        "derivable_direct_std": deriv_direct_s,
        "transient_consumption_mean": trans_m,
        "transient_consumption_std": trans_s,
        "corrupted_chain_sum": sum(r.get("corrupted_chain", 0.0) for r in rows),
        "incorrect_sum": sum(r.get("incorrect", 0.0) for r in rows),
        "incorrect_with_chain_sum": sum(
            r.get("incorrect_with_chain", 0.0) for r in rows
        ),
        "errors_cwa_sum": sum(r.get("errors_cwa", 0.0) for r in rows),
        "errors_chain_linked_sum": sum(r.get("errors_chain_linked", 0.0) for r in rows),
        "errors_direct_sum": sum(r.get("errors_direct", 0.0) for r in rows),
        "derivable_total_sum": sum(r.get("derivable_total", 0.0) for r in rows),
        "derivable_correct_sum": sum(r.get("derivable_correct", 0.0) for r in rows),
        "derivable_incorrect_sum": sum(r.get("derivable_incorrect", 0.0) for r in rows),
        "transient_blocked_sum": sum(r.get("transient_blocked_total", 0.0) for r in rows),
        "transient_consumed_sum": sum(r.get("transient_consumed_total", 0.0) for r in rows),
        "schema_total_sum": sum(r.get("schema_total", 0.0) for r in rows),
        "removed_total_sum": sum(r.get("removed_total", 0.0) for r in rows),
        "reliable_total_sum": sum(r.get("reliable_total", 0.0) for r in rows),
        "reliable_removed_sum": sum(r.get("reliable_removed", 0.0) for r in rows),
        "default_decision_total_sum": sum(
            r.get("default_decision_total", 0.0) for r in rows
        ),
    }


def summarize_problems(problems: List[Problem]) -> Dict[str, object]:
    schemas_per_problem = [len(p.schemas) for p in problems]
    facts_per_problem = [len(p.facts) for p in problems]
    antecedents_per_schema: List[int] = []
    multi_antecedent_count = 0

    depth_hist: Dict[str, int] = {}
    family_counts: Dict[str, int] = {fam: 0 for fam in FAMILIES}
    corrupted_family_counts: Dict[str, int] = {fam: 0 for fam in FAMILIES}
    total_schemas = 0
    total_corrupted = 0
    schemas_by_depth: Dict[int, List[int]] = {}

    for p in problems:
        depth_key = str(p.depth)
        depth_hist[depth_key] = depth_hist.get(depth_key, 0) + 1
        schemas_by_depth.setdefault(p.depth, []).append(len(p.schemas))
        for s in p.schemas:
            total_schemas += 1
            ant_len = len(s.antecedents)
            antecedents_per_schema.append(ant_len)
            if ant_len >= 2:
                multi_antecedent_count += 1
            family_counts[s.family] = family_counts.get(s.family, 0) + 1
            if s.corrupted:
                total_corrupted += 1
                corrupted_family_counts[s.family] = (
                    corrupted_family_counts.get(s.family, 0) + 1
                )

    def safe_mean(vals: List[int]) -> float:
        return statistics.mean(vals) if vals else 0.0

    def safe_std(vals: List[int]) -> float:
        return statistics.pstdev(vals) if len(vals) > 1 else 0.0

    family_ratios = {
        fam: (family_counts.get(fam, 0) / max(1, total_schemas)) for fam in FAMILIES
    }
    corruption_rates_by_family = {
        fam: (
            corrupted_family_counts.get(fam, 0)
            / max(1, family_counts.get(fam, 0))
        )
        for fam in FAMILIES
    }
    schemas_per_depth_mean = {
        str(depth): (statistics.mean(vals) if vals else 0.0)
        for depth, vals in schemas_by_depth.items()
    }
    schemas_per_depth_std = {
        str(depth): (statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        for depth, vals in schemas_by_depth.items()
    }

    return {
        "num_problems": len(problems),
        "num_schemas": total_schemas,
        "num_corrupted": total_corrupted,
        "overall_corruption_rate": total_corrupted / max(1, total_schemas),
        "schemas_per_problem_mean": safe_mean(schemas_per_problem),
        "schemas_per_problem_std": safe_std(schemas_per_problem),
        "schemas_per_problem_min": min(schemas_per_problem) if schemas_per_problem else 0,
        "schemas_per_problem_max": max(schemas_per_problem) if schemas_per_problem else 0,
        "facts_per_problem_mean": safe_mean(facts_per_problem),
        "facts_per_problem_std": safe_std(facts_per_problem),
        "facts_per_problem_min": min(facts_per_problem) if facts_per_problem else 0,
        "facts_per_problem_max": max(facts_per_problem) if facts_per_problem else 0,
        "antecedents_per_schema_mean": safe_mean(antecedents_per_schema),
        "antecedents_per_schema_std": safe_std(antecedents_per_schema),
        "antecedents_per_schema_min": min(antecedents_per_schema)
        if antecedents_per_schema
        else 0,
        "antecedents_per_schema_max": max(antecedents_per_schema)
        if antecedents_per_schema
        else 0,
        "multi_antecedent_schema_ratio": multi_antecedent_count / max(1, total_schemas),
        "depth_histogram": depth_hist,
        "schemas_per_depth_mean": schemas_per_depth_mean,
        "schemas_per_depth_std": schemas_per_depth_std,
        "family_counts": family_counts,
        "family_ratios": family_ratios,
        "corrupted_family_counts": corrupted_family_counts,
        "corruption_rates_by_family": corruption_rates_by_family,
    }


def redirect_exposure_count(problem: Problem) -> int:
    """Count schemas whose antecedents consume at least one schema-produced proposition."""
    consequent_counts: Dict[str, int] = {}
    consequent_by_schema: Dict[str, str] = {}
    for schema in problem.schemas:
        cons = normalize_prop(schema.consequent)
        if not cons:
            continue
        consequent_by_schema[schema.schema_id] = cons
        consequent_counts[cons] = consequent_counts.get(cons, 0) + 1

    exposed = 0
    for schema in problem.schemas:
        schema_cons = consequent_by_schema.get(schema.schema_id, "")
        has_exposure = False
        for ant in schema.antecedents:
            ant_n = normalize_prop(ant)
            if not ant_n:
                continue
            c = consequent_counts.get(ant_n, 0)
            if c == 0:
                continue
            # Require "another schema": if the only matching consequent is this
            # schema's own consequent, do not count exposure.
            if c > 1 or ant_n != schema_cons:
                has_exposure = True
                break
        if has_exposure:
            exposed += 1
    return exposed


def compute_dataset_connectivity_analysis(dataset_results: Dict[str, object]) -> Dict[str, object]:
    datasets = dataset_results.get("datasets", {})
    if not isinstance(datasets, dict):
        datasets = {}
    clean_by_seed = datasets.get("clean", {})
    if not isinstance(clean_by_seed, dict):
        clean_by_seed = {}

    exposure_mean_per_seed: List[float] = []
    exposure_ratio_per_seed: List[float] = []
    schema_count_per_seed: List[float] = []

    for probs in clean_by_seed.values():
        if not isinstance(probs, list):
            continue
        problem_exposures: List[float] = []
        problem_exposure_ratios: List[float] = []
        schema_total = 0.0
        for p in probs:
            if not isinstance(p, Problem):
                continue
            exposure = float(redirect_exposure_count(p))
            schema_n = float(len(p.schemas))
            problem_exposures.append(exposure)
            problem_exposure_ratios.append(exposure / max(1.0, schema_n))
            schema_total += schema_n
        if problem_exposures:
            exposure_mean_per_seed.append(statistics.mean(problem_exposures))
            exposure_ratio_per_seed.append(statistics.mean(problem_exposure_ratios))
            schema_count_per_seed.append(schema_total)

    exposure_mean, exposure_std = mean_std(exposure_mean_per_seed)
    exposure_ratio_mean, exposure_ratio_std = mean_std(exposure_ratio_per_seed)
    schema_mean, schema_std = mean_std(schema_count_per_seed)

    conditions = dataset_results.get("conditions", {})
    if not isinstance(conditions, dict):
        conditions = {}
    redirect_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond in ("random_redirect", "clustered_redirect"):
        cond_rows = conditions.get(cond, {})
        if not isinstance(cond_rows, dict):
            continue
        redirect_stats[cond] = {}
        for model in ("instance_weighted", "aspic_v1_terminal", "aspic_v2", "schema_gating"):
            row = cond_rows.get(model, {})
            if not isinstance(row, dict):
                continue
            redirect_stats[cond][model] = {
                "chain_error_mean": float(row.get("chain_error_mean", 0.0)),
                "chain_error_std": float(row.get("chain_error_std", 0.0)),
                "cascade_mean": float(row.get("cascade_mean", 0.0)),
                "cascade_std": float(row.get("cascade_std", 0.0)),
                "accuracy_mean": float(row.get("accuracy_mean", 0.0)),
                "accuracy_std": float(row.get("accuracy_std", 0.0)),
            }

    gaps: Dict[str, Dict[str, float]] = {}
    for cond, model_rows in redirect_stats.items():
        iw = model_rows.get("instance_weighted", {}).get("chain_error_mean", 0.0)
        v2 = model_rows.get("aspic_v2", {}).get("chain_error_mean", 0.0)
        gating = model_rows.get("schema_gating", {}).get("chain_error_mean", 0.0)
        gaps[cond] = {
            "iw_minus_gating": iw - gating,
            "iw_minus_v2": iw - v2,
            "v2_minus_gating": v2 - gating,
        }

    return {
        "redirect_exposure_mean": exposure_mean,
        "redirect_exposure_std": exposure_std,
        "redirect_exposure_ratio_mean": exposure_ratio_mean,
        "redirect_exposure_ratio_std": exposure_ratio_std,
        "schema_count_per_seed_mean": schema_mean,
        "schema_count_per_seed_std": schema_std,
        "redirect": redirect_stats,
        "redirect_gaps": gaps,
    }


def quality_flags_for_summary(summary: Dict[str, object]) -> List[str]:
    flags: List[str] = []
    spp_mean = float(summary.get("schemas_per_problem_mean", 0.0))
    if spp_mean < 3.0:
        flags.append("low_schema_density_overall")

    depth_mean = summary.get("schemas_per_depth_mean", {})
    if isinstance(depth_mean, dict):
        d5 = float(depth_mean.get("5", 0.0))
        if d5 > 0.0 and d5 < 3.0:
            flags.append("low_schema_density_depth5")

    fam_ratios = summary.get("family_ratios", {})
    if isinstance(fam_ratios, dict):
        imp = float(fam_ratios.get("IMPLICATION", 0.0))
        dfl = float(fam_ratios.get("DEFAULT", 0.0))
        if imp >= 0.90:
            flags.append("family_distribution_degenerate_implication_dominant")
        if dfl <= 0.05:
            flags.append("family_distribution_default_too_rare")

    return flags


def build_matching_diagnostics(
    problems: List[Problem], sample_count: int = 10
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    total_ants = 0
    ants_in_facts = 0
    q_in_facts = 0
    q_in_any_consequent = 0

    for p in problems[:sample_count]:
        fact_set = set(p.facts)
        entities = sorted({ent for fact in fact_set for ent in prop_entities(fact)})
        schemas_view: List[Dict[str, object]] = []

        for s in p.schemas:
            ant_view: List[Dict[str, object]] = []
            for ant in s.antecedents:
                if prop_has_variable(ant):
                    bindings = candidate_bindings_for_pattern(ant, fact_set)
                    candidates = [substitute_variable(ant, b) for b in bindings]
                else:
                    candidates = [ant]
                match = any(c in fact_set for c in candidates)
                total_ants += 1
                if match:
                    ants_in_facts += 1
                ant_view.append(
                    {
                        "antecedent": ant,
                        "in_fact_set": match,
                        "candidate_count": len(candidates),
                    }
                )

            schemas_view.append(
                {
                    "schema_id": s.schema_id,
                    "family": s.family,
                    "text": s.text,
                    "antecedents": ant_view,
                    "consequent": s.consequent,
                }
            )

        q = normalize_prop(p.question)
        q_is_fact = q in fact_set
        q_is_consequent = False
        for s in p.schemas:
            if prop_has_variable(s.consequent):
                bindings = entities
                candidates = [substitute_variable(s.consequent, b) for b in bindings]
            else:
                candidates = [s.consequent]
            if q in candidates:
                q_is_consequent = True
                break
        if q_is_fact:
            q_in_facts += 1
        if q_is_consequent:
            q_in_any_consequent += 1

        rows.append(
            {
                "problem_id": p.problem_id,
                "depth": p.depth,
                "question": q,
                "answer": p.answer,
                "facts": sorted(fact_set),
                "question_in_fact_set": q_is_fact,
                "question_in_any_schema_consequent": q_is_consequent,
                "schemas": schemas_view,
            }
        )

    return {
        "sample_count": len(rows),
        "antecedent_match_rate_in_initial_facts": ants_in_facts / max(1, total_ants),
        "question_in_initial_facts_rate": q_in_facts / max(1, len(rows)),
        "question_in_any_schema_consequent_rate": q_in_any_consequent / max(1, len(rows)),
        "rows": rows,
    }


def run_dataset_experiments(
    cfg: Dict[str, object],
    dataset_key: str,
    default_policy_override: Optional[str] = None,
) -> Dict[str, object]:
    seeds = [int(s) for s in cfg["seeds"]]
    run_cfg = copy.deepcopy(cfg)
    if default_policy_override is not None:
        reason_cfg = run_cfg.get("reasoner")
        if not isinstance(reason_cfg, dict):
            reason_cfg = {}
        reason_cfg["default_policy"] = default_policy_override
        run_cfg["reasoner"] = reason_cfg

    base = load_problems_base(run_cfg, dataset_key=dataset_key)

    # Build per-condition/seed datasets from same base sample.
    datasets: Dict[str, Dict[int, List[Problem]]] = {cond: {} for cond in CONDITIONS}
    for cond in CONDITIONS:
        for seed in seeds:
            datasets[cond][seed] = apply_noise(base, cond, seed, run_cfg)

    per_condition: Dict[str, object] = {}
    clean_fnr_vals: List[float] = []
    clustered_reduction_vals: List[float] = []
    estimator_metrics: Dict[str, Dict[int, Dict[str, float]]] = {cond: {} for cond in CONDITIONS}

    for cond in CONDITIONS:
        per_model_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in MODELS}

        for seed in seeds:
            probs = datasets[cond][seed]
            records = build_schema_records(probs)
            estimator = train_reliability_estimator(
                records=records,
                cfg=run_cfg,
                seed=seed,
                label_noise_rate=0.0,
            )
            estimator_metrics[cond][seed] = estimator.metrics

            reliable_sources = source_reliability_map(probs)

            for model in MODELS:
                row = evaluate_model_on_condition(
                    problems=probs,
                    estimator=estimator,
                    model=model,
                    cfg=run_cfg,
                    seed=seed,
                    reliable_sources=reliable_sources,
                )
                per_model_rows[model].append(row)

                if cond == "clean" and model == "schema_gating":
                    fnr = row["reliable_removed"] / max(1.0, row["reliable_total"])
                    clean_fnr_vals.append(fnr)

                if cond.startswith("clustered") and model == "schema_gating":
                    reduction = row["removed_total"] / max(1.0, row["schema_total"])
                    clustered_reduction_vals.append(reduction)

        per_condition[cond] = {
            model: aggregate_seed_rows(rows) for model, rows in per_model_rows.items()
        }

    return {
        "conditions": per_condition,
        "dataset_key": dataset_key,
        "clean_false_negative_rate": statistics.mean(clean_fnr_vals) if clean_fnr_vals else 0.0,
        "clustered_schema_reduction": statistics.mean(clustered_reduction_vals)
        if clustered_reduction_vals
        else 0.0,
        "estimator_metrics": estimator_metrics,
        "datasets": datasets,
        "default_policy": str(run_cfg.get("reasoner", {}).get("default_policy", "closed_world")),
    }


def run_ruletaker_experiments(cfg: Dict[str, object]) -> Dict[str, object]:
    return run_dataset_experiments(cfg, dataset_key="ruletaker")


def run_proofwriter_experiments(cfg: Dict[str, object]) -> Dict[str, object]:
    pw_cfg = cfg.get("proofwriter", {})
    if not isinstance(pw_cfg, dict):
        raise ValueError("`proofwriter` config block must be an object.")
    default_policy = str(pw_cfg.get("default_policy", "random_tie"))
    return run_dataset_experiments(
        cfg,
        dataset_key="proofwriter",
        default_policy_override=default_policy,
    )


def run_degraded_ablation(
    cfg: Dict[str, object],
    datasets_by_condition: Dict[str, Dict[int, List[Problem]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    ablation_cfg = cfg.get("ablation", {})
    if not isinstance(ablation_cfg, dict):
        ablation_cfg = {}

    target_model = str(ablation_cfg.get("target_model", "schema_gating"))
    reference_model = str(ablation_cfg.get("reference_model", "instance_weighted"))
    metric = str(ablation_cfg.get("metric", "cascade_rate"))
    eval_conditions = ablation_cfg.get(
        "conditions",
        ["clustered_negate", "clustered_redirect"],
    )
    if not isinstance(eval_conditions, list):
        eval_conditions = [str(eval_conditions)]
    eval_conditions = [str(c) for c in eval_conditions if str(c) in datasets_by_condition]
    if not eval_conditions:
        eval_conditions = [c for c in ("clustered_negate", "clustered_redirect") if c in datasets_by_condition]

    valid_metrics = {"cascade_rate", "chain_error_rate"}
    if target_model not in MODELS:
        raise ValueError(f"Unsupported ablation target model: {target_model}")
    if reference_model not in MODELS:
        raise ValueError(f"Unsupported ablation reference model: {reference_model}")
    if metric not in valid_metrics:
        raise ValueError(
            f"Unsupported ablation metric: {metric}. Use one of {sorted(valid_metrics)}."
        )

    rates = [float(r) for r in cfg["estimator"]["label_noise_rates"]]
    seeds = [int(s) for s in cfg["seeds"]]

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond in eval_conditions:
        cond_datasets = datasets_by_condition[cond]
        cond_out: Dict[str, Dict[str, float]] = {}
        for rate in rates:
            deltas: List[float] = []
            for seed in seeds:
                probs = cond_datasets[seed]
                records = build_schema_records(probs)
                estimator = train_reliability_estimator(
                    records=records,
                    cfg=cfg,
                    seed=seed,
                    label_noise_rate=rate,
                )
                reliable_sources = source_reliability_map(probs)

                ref_row = evaluate_model_on_condition(
                    problems=probs,
                    estimator=estimator,
                    model=reference_model,
                    cfg=cfg,
                    seed=seed,
                    reliable_sources=reliable_sources,
                )
                target_row = evaluate_model_on_condition(
                    problems=probs,
                    estimator=estimator,
                    model=target_model,
                    cfg=cfg,
                    seed=seed,
                    reliable_sources=reliable_sources,
                )
                deltas.append(float(target_row[metric]) - float(ref_row[metric]))

            mean_d, std_d = mean_std(deltas)
            cond_out[f"{int(rate * 100)}%"] = {"delta_mean": mean_d, "delta_std": std_d}
        out[cond] = cond_out
    return out


def generate_synthetic_redirect_problems(
    n_problems: int,
    seed: int,
    corruption_model: str,
    chain_length: int = 3,
    corrupt_rate: float = 1.0,
) -> List[Problem]:
    corruption_model = corruption_model.strip().lower()
    if corruption_model not in ("redirect", "negation"):
        raise ValueError(
            f"Unsupported corruption_model `{corruption_model}`. Use `redirect` or `negation`."
        )
    if chain_length < 3:
        raise ValueError("chain_length must be >= 3.")

    rng = random.Random(deterministic_seed("synthetic_redirect", seed, corruption_model, chain_length))
    problems: List[Problem] = []

    for i in range(n_problems):
        entity = f"obj{i:04d}"
        pid = f"synthetic_{corruption_model}_L{chain_length}_{seed}_{i:04d}"
        fact_a = f"u|{entity}|prop_a"
        fact_b = f"u|{entity}|prop_b"
        m_props = [f"u|{entity}|derived_m_{j}" for j in range(chain_length - 1)]
        m_start = m_props[0]
        m_final = m_props[-1]
        m_wrong = f"u|{entity}|derived_m_wrong"
        q_prop = f"u|{entity}|answer"
        not_q = negate_prop(q_prop)

        s1 = Schema(
            schema_id=f"syn_s1_{pid}",
            text=f"if prop_a and prop_b then derived_m_0 for {entity}",
            antecedents=(fact_a, fact_b),
            consequent=m_start,
            family="TRANSITIVE",
            source_id="src_A_synthetic",
            corrupted=False,
        )

        schemas: List[Schema] = [s1]
        for j in range(1, chain_length - 1):
            src = m_props[j - 1]
            dst = m_props[j]
            schemas.append(
                Schema(
                    schema_id=f"syn_mid{j}_{pid}",
                    text=f"if derived_m_{j - 1} then derived_m_{j} for {entity}",
                    antecedents=(src,),
                    consequent=dst,
                    family="IMPLICATION",
                    source_id="src_B_synthetic",
                    corrupted=False,
                )
            )

        schemas.append(
            Schema(
                schema_id=f"syn_terminal_{pid}",
                text=f"if derived_m_{chain_length - 2} then answer for {entity}",
                antecedents=(m_final,),
                consequent=q_prop,
                family="IMPLICATION",
                source_id="src_B_synthetic",
                corrupted=False,
            )
        )
        schemas.append(
            Schema(
                schema_id=f"syn_cascade_{pid}",
                text=f"if derived_m_wrong then not_answer for {entity}",
                antecedents=(m_wrong,),
                consequent=not_q,
                family="IMPLICATION",
                source_id="src_B_synthetic",
                corrupted=False,
            )
        )

        if rng.random() < corrupt_rate:
            if corruption_model == "redirect":
                corrupt_schema_redirect(s1, m_wrong)
            else:
                corrupt_schema(s1)

        problems.append(
            Problem(
                problem_id=pid,
                depth=2,
                question=q_prop,
                answer=1,
                facts=(fact_a, fact_b),
                schemas=schemas,
            )
        )

    return problems


def run_synthetic_redirect_experiment(
    cfg: Dict[str, object],
    n_problems: int = 60,
) -> Dict[str, object]:
    """Redirect-vs-negation corruption comparison + redirect chain-length scaling."""
    seeds = [int(s) for s in cfg["seeds"]]
    accumulator: Dict[str, Dict[str, List[float]]] = {
        cm: {m: [] for m in MODELS} for cm in ("redirect", "negation")
    }
    synth_cfg = cfg.get("synthetic_redirect", {})
    if not isinstance(synth_cfg, dict):
        synth_cfg = {}
    chain_lengths = synth_cfg.get("chain_lengths", [3, 4, 5])
    if not isinstance(chain_lengths, list):
        chain_lengths = [3, 4, 5]
    parsed_chain_lengths: List[int] = []
    for x in chain_lengths:
        try:
            xv = int(x)
        except Exception:
            continue
        if xv >= 3:
            parsed_chain_lengths.append(xv)
    if not parsed_chain_lengths:
        parsed_chain_lengths = [3, 4, 5]
    chain_accumulator: Dict[int, Dict[str, List[float]]] = {
        cl: {m: [] for m in MODELS} for cl in parsed_chain_lengths
    }

    perfect_estimator = EstimatorModel(
        threshold=0.5,
        backend="perfect_graded_synthetic",
        source_scores={"src_A_synthetic": 0.3, "src_B_synthetic": 0.8},
        family_scores={},
        global_score=0.55,
        metrics={"val_acc": 1.0, "test_acc": 1.0},
    )
    reliable_sources = {"src_A_synthetic": 0.0, "src_B_synthetic": 1.0}

    for seed in seeds:
        for corruption_model in ("redirect", "negation"):
            problems = generate_synthetic_redirect_problems(
                n_problems=n_problems,
                seed=seed,
                corruption_model=corruption_model,
                chain_length=3,
                corrupt_rate=1.0,
            )
            for model in MODELS:
                row = evaluate_model_on_condition(
                    problems=problems,
                    estimator=perfect_estimator,
                    model=model,
                    cfg=cfg,
                    seed=seed,
                    reliable_sources=reliable_sources,
                )
                accumulator[corruption_model][model].append(float(row["chain_error_rate"]))

        for chain_length in parsed_chain_lengths:
            problems = generate_synthetic_redirect_problems(
                n_problems=n_problems,
                seed=seed,
                corruption_model="redirect",
                chain_length=chain_length,
                corrupt_rate=1.0,
            )
            for model in MODELS:
                row = evaluate_model_on_condition(
                    problems=problems,
                    estimator=perfect_estimator,
                    model=model,
                    cfg=cfg,
                    seed=seed,
                    reliable_sources=reliable_sources,
                )
                chain_accumulator[chain_length][model].append(float(row["chain_error_rate"]))

    out: Dict[str, object] = {}
    for corruption_model in ("redirect", "negation"):
        out[corruption_model] = {}
        for model in MODELS:
            vals = accumulator[corruption_model][model]
            m, s = mean_std(vals)
            out[corruption_model][model] = {"chain_error_mean": m, "chain_error_std": s}
    out["chain_lengths"] = parsed_chain_lengths
    out["redirect_by_chain_length"] = {}
    for cl in parsed_chain_lengths:
        cl_key = str(cl)
        out["redirect_by_chain_length"][cl_key] = {}
        for model in MODELS:
            vals = chain_accumulator[cl][model]
            m, s = mean_std(vals)
            out["redirect_by_chain_length"][cl_key][model] = {
                "chain_error_mean": m,
                "chain_error_std": s,
            }
    return out


# -----------------------------
# Report
# -----------------------------


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def build_report(
    hardware: Dict[str, object],
    ruletaker: Dict[str, object],
    ablation: Dict[str, Dict[str, Dict[str, float]]],
    cfg: Dict[str, object],
    synthetic_redirect: Optional[Dict[str, object]] = None,
    proofwriter: Optional[Dict[str, object]] = None,
    ablation_dataset_key: Optional[str] = None,
) -> str:
    seeds_n = len(cfg.get("seeds", [])) if isinstance(cfg.get("seeds"), list) else 0
    model_label = {
        "aspic_v2_persistent": "aspic_v2_persistent",
    }
    base_conditions = ("random", "clustered")

    lines: List[str] = []
    lines.append("# Experiment Report")
    lines.append("")

    lines.append("## Hardware")
    lines.append(f"- Platform: `{hardware.get('platform')}`")
    lines.append(f"- CPU logical cores: `{hardware.get('cpu_logical_count')}`")
    lines.append(f"- RAM total: `{hardware.get('memory_total_gib')} GiB`")
    lines.append(f"- RAM available at run start: `{hardware.get('memory_available_gib')} GiB`")
    lines.append(f"- Disk free: `{hardware.get('disk_free_gib')} GiB`")
    gpus = hardware.get("gpus", [])
    if gpus and isinstance(gpus, list) and isinstance(gpus[0], dict) and "name" in gpus[0]:
        lines.append(
            f"- GPU: `{gpus[0]['name']}` | VRAM: `{gpus[0]['memory_total']}` | Driver: `{gpus[0]['driver_version']}`"
        )
    else:
        lines.append("- GPU: not available to `nvidia-smi` in this run context")
    lines.append("")

    def append_dataset_section(
        title: str,
        dataset_key: str,
        dataset_results: Dict[str, object],
    ) -> None:
        conditions = dataset_results.get("conditions", {})
        if not isinstance(conditions, dict):
            return

        per_seed_n = int(cfg.get(dataset_key, {}).get("num_problems", 0))
        total_eval = max(1, seeds_n * per_seed_n)

        lines.append(f"## {title} Main Results")
        lines.append("### Clean")
        lines.append(
            "| Model | Accuracy (mean±std) | Cascade (mean±std) | Chain-Linked Error (mean±std) | V2 Transient Use (mean±std) |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        clean_rows = conditions.get("clean", {})
        if not isinstance(clean_rows, dict):
            clean_rows = {}
        for model in MODELS:
            row = clean_rows.get(model, {})
            if not isinstance(row, dict):
                continue
            transient_summary = (
                f"{float(row.get('transient_consumption_mean', 0.0)):.3f} ± {float(row.get('transient_consumption_std', 0.0)):.3f}"
                if model.startswith("aspic_v2")
                else "-"
            )
            lines.append(
                f"| {model_label.get(model, model)} | {float(row.get('accuracy_mean', 0.0)):.3f} ± {float(row.get('accuracy_std', 0.0)):.3f} | {float(row.get('cascade_mean', 0.0)):.3f} ± {float(row.get('cascade_std', 0.0)):.3f} | {float(row.get('chain_error_mean', 0.0)):.3f} ± {float(row.get('chain_error_std', 0.0)):.3f} | {transient_summary} |"
            )
        lines.append("")

        for base in base_conditions:
            cond_neg = f"{base}_negate"
            cond_red = f"{base}_redirect"
            rows_neg = conditions.get(cond_neg, {})
            rows_red = conditions.get(cond_red, {})
            if not rows_neg and not rows_red:
                continue
            lines.append(f"### {base.title()} (Negate vs Redirect)")
            lines.append(
                "| Model | Negate Acc | Negate Cascade | Negate Chain | Redirect Acc | Redirect Cascade | Redirect Chain |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for model in MODELS:
                rn = rows_neg.get(model, {}) if isinstance(rows_neg, dict) else {}
                rr = rows_red.get(model, {}) if isinstance(rows_red, dict) else {}
                if not isinstance(rn, dict):
                    rn = {}
                if not isinstance(rr, dict):
                    rr = {}
                lines.append(
                    f"| {model_label.get(model, model)} | "
                    f"{float(rn.get('accuracy_mean', 0.0)):.3f} | {float(rn.get('cascade_mean', 0.0)):.3f} | {float(rn.get('chain_error_mean', 0.0)):.3f} | "
                    f"{float(rr.get('accuracy_mean', 0.0)):.3f} | {float(rr.get('cascade_mean', 0.0)):.3f} | {float(rr.get('chain_error_mean', 0.0)):.3f} |"
                )
            lines.append("")

        lines.append("")
        lines.append(
            "- `Cascade` is defined as `corrupted_chain / incorrect` (paper definition)."
        )
        lines.append(
            "- `Chain-Linked Error` is `incorrect_with_chain / incorrect` (diagnostic coverage metric)."
        )
        lines.append(
            "- `aspic_v2_persistent` is the one-step persistent chain-intercepting variant."
        )
        lines.append(
            "- It keeps blocked transient nodes visible for one extra step; under redirect corruption this can approach instance-weighted behavior via cross-step consumption."
        )
        lines.append("")

        lines.append(f"## {title} Mechanism Checks")
        lines.append(
            f"- Clean false-negative rate (reliable schemas removed): `{format_pct(float(dataset_results.get('clean_false_negative_rate', 0.0)))}`"
        )
        lines.append(
            f"- Clustered schema construction reduction (gating): `{format_pct(float(dataset_results.get('clustered_schema_reduction', 0.0)))}`"
        )
        default_policy = str(
            dataset_results.get(
                "default_policy",
                cfg.get("reasoner", {}).get("default_policy", "closed_world"),
            )
        )
        lines.append(
            f"- Low-confidence fallback decision rates (policy=`{default_policy}`):"
        )
        for cond in CONDITIONS:
            cond_rows = conditions.get(cond, {})
            if not isinstance(cond_rows, dict):
                continue
            for model in MODELS:
                row = cond_rows.get(model, {})
                if not isinstance(row, dict):
                    continue
                default_rate = float(row.get("default_decision_total_sum", 0.0)) / total_eval
                lines.append(
                    f"  - `{cond}/{model_label.get(model, model)}`: `{format_pct(default_rate)}`"
                )
        lines.append("- ASPIC+ transient propagation window usage:")
        for cond in CONDITIONS:
            cond_rows = conditions.get(cond, {})
            if not isinstance(cond_rows, dict):
                continue
            for model in MODELS:
                if not model.startswith("aspic_v2"):
                    continue
                row = cond_rows.get(model, {})
                if not isinstance(row, dict):
                    continue
                blocked = float(row.get("transient_blocked_sum", 0.0))
                consumed = float(row.get("transient_consumed_sum", 0.0))
                rate = consumed / max(1.0, blocked)
                lines.append(
                    f"  - `{cond}/{model_label.get(model, model)}`: blocked `{int(blocked)}`, consumed `{int(consumed)}`, utilization `{format_pct(rate)}`"
                )
        lines.append("")

        lines.append(f"## {title} Error Decomposition")
        lines.append("| Condition | Model | CWA Error | Chain-Linked Error | Direct Error |")
        lines.append("|---|---|---:|---:|---:|")
        for cond in CONDITIONS:
            cond_rows = conditions.get(cond, {})
            if not isinstance(cond_rows, dict):
                continue
            for model in MODELS:
                row = cond_rows.get(model, {})
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {cond} | {model_label.get(model, model)} | "
                    f"{float(row.get('error_cwa_mean', 0.0)):.3f} ± {float(row.get('error_cwa_std', 0.0)):.3f} | "
                    f"{float(row.get('error_chain_linked_mean', 0.0)):.3f} ± {float(row.get('error_chain_linked_std', 0.0)):.3f} | "
                    f"{float(row.get('error_direct_mean', 0.0)):.3f} ± {float(row.get('error_direct_std', 0.0)):.3f} |"
                )
        lines.append("")

        lines.append(f"## {title} Derivable-Only Metrics")
        lines.append(
            "| Condition | Model | Derivable Coverage | Derivable Accuracy | Derivable Chain-Linked Error |"
        )
        lines.append("|---|---|---:|---:|---:|")
        for cond in CONDITIONS:
            cond_rows = conditions.get(cond, {})
            if not isinstance(cond_rows, dict):
                continue
            for model in MODELS:
                row = cond_rows.get(model, {})
                if not isinstance(row, dict):
                    continue
                coverage = float(row.get("derivable_total_sum", 0.0)) / total_eval
                lines.append(
                    f"| {cond} | {model_label.get(model, model)} | {coverage:.3f} | "
                    f"{float(row.get('derivable_accuracy_mean', 0.0)):.3f} ± {float(row.get('derivable_accuracy_std', 0.0)):.3f} | "
                    f"{float(row.get('derivable_chain_linked_mean', 0.0)):.3f} ± {float(row.get('derivable_chain_linked_std', 0.0)):.3f} |"
                )
        lines.append("")

    if ruletaker and isinstance(ruletaker, dict) and isinstance(ruletaker.get("conditions"), dict):
        append_dataset_section("RuleTaker", "ruletaker", ruletaker)
    if proofwriter and isinstance(proofwriter, dict):
        append_dataset_section("ProofWriter", "proofwriter", proofwriter)

    connectivity_rows: List[Tuple[str, Dict[str, object]]] = []
    if ruletaker and isinstance(ruletaker, dict):
        ca = ruletaker.get("connectivity_analysis", {})
        if isinstance(ca, dict):
            connectivity_rows.append(("RuleTaker", ca))
    if proofwriter and isinstance(proofwriter, dict):
        ca = proofwriter.get("connectivity_analysis", {})
        if isinstance(ca, dict):
            connectivity_rows.append(("ProofWriter", ca))

    if connectivity_rows:
        lines.append("## Redirect Exposure Analysis")
        lines.append(
            "| Dataset | Redirect exposure (schemas/problem) | Redirect exposure ratio | Clustered redirect gap (IW-gating) |"
        )
        lines.append("|---|---:|---:|---:|")
        for dataset_name, ca in connectivity_rows:
            clustered_gap = float(
                ca.get("redirect_gaps", {})
                .get("clustered_redirect", {})
                .get("iw_minus_gating", 0.0)
            )
            lines.append(
                f"| {dataset_name} | "
                f"{float(ca.get('redirect_exposure_mean', 0.0)):.3f} ± {float(ca.get('redirect_exposure_std', 0.0)):.3f} | "
                f"{float(ca.get('redirect_exposure_ratio_mean', 0.0)):.3f} ± {float(ca.get('redirect_exposure_ratio_std', 0.0)):.3f} | "
                f"{clustered_gap:.3f} |"
            )
        lines.append("")

    if len(connectivity_rows) >= 2:
        conn_map = {name: ca for name, ca in connectivity_rows}
        rt = conn_map.get("RuleTaker", {})
        pw = conn_map.get("ProofWriter", {})
        if isinstance(rt, dict) and isinstance(pw, dict):
            rt_exp = float(rt.get("redirect_exposure_mean", 0.0))
            pw_exp = float(pw.get("redirect_exposure_mean", 0.0))
            lines.append("## Discussion Note")
            if pw_exp < rt_exp:
                lines.append(
                    f"ProofWriter shows lower redirect exposure than RuleTaker (ProofWriter: {pw_exp:.2f}, RuleTaker: {rt_exp:.2f}), "
                    "which is consistent with the smaller redirect effect observed on ProofWriter."
                )
            else:
                lines.append(
                    f"ProofWriter does not show lower redirect exposure than RuleTaker (ProofWriter: {pw_exp:.2f}, RuleTaker: {rt_exp:.2f}), "
                    "so redirect exposure alone does not explain the smaller ProofWriter redirect effect."
                )
            lines.append("")

    ablation_cfg = cfg.get("ablation", {})
    if not isinstance(ablation_cfg, dict):
        ablation_cfg = {}
    ablation_target_model = str(ablation_cfg.get("target_model", "schema_gating"))
    ablation_reference_model = str(ablation_cfg.get("reference_model", "instance_weighted"))
    ablation_metric = str(ablation_cfg.get("metric", "cascade_rate"))

    lines.append("## Degraded Estimator Ablation")
    if ablation_dataset_key:
        lines.append(f"- Dataset: `{ablation_dataset_key}`")
    metric_label = (
        "Chain-Linked Error"
        if ablation_metric == "chain_error_rate"
        else "Cascade"
    )
    if not ablation:
        lines.append("- Not run for this report.")
        lines.append("")
    for cond in sorted(ablation.keys()):
        cond_rows = ablation.get(cond, {})
        if not isinstance(cond_rows, dict):
            continue
        lines.append(f"### {cond}")
        lines.append(
            f"| Label noise | Δ {metric_label} ({model_label.get(ablation_target_model, ablation_target_model)} - {model_label.get(ablation_reference_model, ablation_reference_model)}) mean±std |"
        )
        lines.append("|---|---:|")
        for key in sorted(cond_rows.keys(), key=lambda x: int(str(x).rstrip('%'))):
            row = cond_rows[key]
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {key} | {float(row.get('delta_mean', 0.0)):.3f} ± {float(row.get('delta_std', 0.0)):.3f} |"
            )
        lines.append("")
    lines.append("")

    lines.append("## Estimator Notes")
    backend = cfg["estimator"].get("backend", "count_fallback")
    lines.append(f"- Configured estimator backend: `{backend}`")
    metric_paths: List[str] = []
    if ruletaker and isinstance(ruletaker, dict):
        metric_paths.append("`ruletaker.estimator_metrics`")
    if proofwriter and isinstance(proofwriter, dict):
        metric_paths.append("`proofwriter.estimator_metrics`")
    lines.append(
        "- Per-condition seed metrics saved in `results.json` under "
        + ", ".join(metric_paths)
        + "."
    )

    if synthetic_redirect and isinstance(synthetic_redirect, dict):
        synth_cfg = cfg.get("synthetic_redirect", {})
        if not isinstance(synth_cfg, dict):
            synth_cfg = {}
        n_synthetic = int(synth_cfg.get("n_problems", 60))
        n_seeds = len(cfg.get("seeds", [])) if isinstance(cfg.get("seeds"), list) else 0

        lines.append("")
        lines.append("## Synthetic Redirect-Corruption Experiment")
        lines.append(
            "Three-schema synthetic problems. S1 (TRANSITIVE, corruptible src_A), "
            "S2 (IMPLICATION, reliable src_B): M->Q, S3 (IMPLICATION, reliable src_B): M_wrong->not Q. "
            "Negation corruption: S1->not M (downstream schemas never fire). "
            "Redirect corruption: S1->M_wrong (S3 fires and commits not Q via cascade). "
            f"Perfect estimator: src_A=0.3, src_B=0.8. {n_synthetic} problems x {n_seeds} seeds."
        )
        lines.append("")
        lines.append("| Model | Negation chain-error | Redirect chain-error | Δ |")
        lines.append("|---|---:|---:|---:|")
        for model in MODELS:
            neg = synthetic_redirect.get("negation", {}).get(model, {})
            red = synthetic_redirect.get("redirect", {}).get(model, {})
            if not isinstance(neg, dict):
                neg = {}
            if not isinstance(red, dict):
                red = {}
            nm = float(neg.get("chain_error_mean", 0.0))
            ns = float(neg.get("chain_error_std", 0.0))
            rm = float(red.get("chain_error_mean", 0.0))
            rs = float(red.get("chain_error_std", 0.0))
            lines.append(
                f"| {model_label.get(model, model)} | {nm:.3f} ± {ns:.3f} | {rm:.3f} ± {rs:.3f} | {rm - nm:+.3f} |"
            )

        chain_lengths = synthetic_redirect.get("chain_lengths", [])
        redirect_by_len = synthetic_redirect.get("redirect_by_chain_length", {})
        if isinstance(chain_lengths, list) and isinstance(redirect_by_len, dict):
            parsed_chain_lengths = [str(int(x)) for x in chain_lengths if str(x).isdigit()]
            if parsed_chain_lengths:
                lines.append("")
                lines.append("### Redirect Chain-Length Scaling")
                lines.append(
                    "- This table is interpreted as stability across chain lengths, not monotonic growth."
                )
                header = "| Model | " + " | ".join(f"L={cl}" for cl in parsed_chain_lengths) + " |"
                sep = "|---|" + "---:|" * len(parsed_chain_lengths)
                lines.append(header)
                lines.append(sep)
                for model in MODELS:
                    vals: List[str] = []
                    for cl in parsed_chain_lengths:
                        row = redirect_by_len.get(cl, {}).get(model, {})
                        if not isinstance(row, dict):
                            row = {}
                        vals.append(
                            f"{float(row.get('chain_error_mean', 0.0)):.3f} ± {float(row.get('chain_error_std', 0.0)):.3f}"
                        )
                    lines.append(
                        f"| {model_label.get(model, model)} | " + " | ".join(vals) + " |"
                    )

    return "\n".join(lines)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper reproduction experiments.")
    parser.add_argument("--config", default="configs/config_both_modes.json")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Override num_problems for each configured dataset block.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast sanity run (single seed, small sample, skips synthetic and ablation).",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip degraded-estimator ablation.",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic redirect experiment.",
    )
    parser.add_argument(
        "--sweep-rates",
        default="",
        help=(
            "Comma-separated clustered corruption rates to sweep (e.g., "
            "0.05,0.10,0.20,0.40,0.60). Runs clustered_redirect table generation."
        ),
    )
    parser.add_argument(
        "--tau-sweep",
        default="",
        help=(
            "Comma-separated estimator thresholds to sweep (e.g., 0.3,0.5,0.7). "
            "Runs full dataset experiments per tau and reports stability."
        ),
    )
    parser.add_argument(
        "--dataset-key",
        default="ruletaker",
        help="Dataset config key for diagnostics (e.g., ruletaker, proofwriter).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Load and preprocess datasets, then exit without training/evaluation.",
    )
    parser.add_argument(
        "--diagnose-matching",
        action="store_true",
        help="Inspect antecedent-to-fact matching on clean problems, then exit.",
    )
    parser.add_argument(
        "--diagnose-samples",
        type=int,
        default=10,
        help="Number of clean problems to include in matching diagnostics.",
    )
    return parser.parse_args()


def parse_sweep_rates(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return []
    rates: List[float] = []
    for p in parts:
        val = float(p)
        if val <= 0.0 or val >= 1.0:
            raise ValueError(f"Invalid sweep rate `{p}`; expected 0.0 < rate < 1.0.")
        rates.append(val)
    # Preserve order while dropping duplicates.
    deduped: List[float] = []
    seen: set[float] = set()
    for r in rates:
        if r in seen:
            continue
        seen.add(r)
        deduped.append(r)
    return deduped


def parse_tau_values(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return []
    taus: List[float] = []
    for p in parts:
        val = float(p)
        if val <= 0.0 or val >= 1.0:
            raise ValueError(f"Invalid tau `{p}`; expected 0.0 < tau < 1.0.")
        taus.append(val)
    deduped: List[float] = []
    seen: set[float] = set()
    for t in taus:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped


def _assert_paper_locked_args(
    args: argparse.Namespace,
    sweep_rates: List[float],
    tau_values: List[float],
) -> None:
    """Enforce canonical paper settings for direct CLI invocation."""
    cfg = Path(args.config).as_posix()
    if cfg != PAPER_CONFIG_PATH.as_posix():
        raise ValueError(
            "Paper-locked mode requires --config configs/config_both_modes.json "
            "(recommended: use ./scripts/reproduce.sh)."
        )
    if args.max_problems is not None:
        raise ValueError("Paper-locked mode forbids --max-problems.")
    if args.smoke_test:
        raise ValueError("Paper-locked mode forbids --smoke-test.")
    if args.prepare_only:
        raise ValueError("Paper-locked mode forbids --prepare-only.")
    if args.diagnose_matching:
        raise ValueError("Paper-locked mode forbids --diagnose-matching.")
    if args.dataset_key != "ruletaker":
        raise ValueError("Paper-locked mode requires --dataset-key ruletaker.")

    outdir = Path(args.outdir).as_posix()
    if sweep_rates:
        if tau_values:
            raise ValueError("Use either --sweep-rates or --tau-sweep.")
        if outdir != PAPER_SWEEP_OUTDIR:
            raise ValueError(f"Paper sweep run requires --outdir {PAPER_SWEEP_OUTDIR}.")
        if sweep_rates != PAPER_SWEEP_RATES:
            raise ValueError(
                "Paper sweep run requires --sweep-rates 0.05,0.10,0.20,0.40,0.60."
            )
        if not args.skip_ablation or not args.skip_synthetic:
            raise ValueError(
                "Paper sweep run requires --skip-ablation and --skip-synthetic."
            )
        return

    if tau_values:
        if outdir != PAPER_TAU_OUTDIR:
            raise ValueError(f"Paper tau run requires --outdir {PAPER_TAU_OUTDIR}.")
        if tau_values != PAPER_TAU_VALUES:
            raise ValueError("Paper tau run requires --tau-sweep 0.3,0.5,0.7.")
        if args.skip_ablation or args.skip_synthetic:
            raise ValueError("Paper tau run forbids --skip-ablation/--skip-synthetic.")
        return

    if outdir != PAPER_MAIN_OUTDIR:
        raise ValueError(f"Paper main run requires --outdir {PAPER_MAIN_OUTDIR}.")
    if args.skip_ablation or args.skip_synthetic:
        raise ValueError("Paper main run forbids --skip-ablation/--skip-synthetic.")


def apply_cli_overrides(cfg: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    out = copy.deepcopy(cfg)
    max_problems = args.max_problems
    if args.smoke_test and max_problems is None:
        max_problems = 30

    if max_problems is not None:
        for key in ("ruletaker", "proofwriter"):
            block = out.get(key)
            if isinstance(block, dict):
                block["num_problems"] = int(max_problems)

    if args.smoke_test:
        seeds = out.get("seeds", [42])
        if isinstance(seeds, list) and seeds:
            out["seeds"] = [int(seeds[0])]

        for key in ("ruletaker", "proofwriter"):
            block = out.get(key)
            if isinstance(block, dict):
                depths = block.get("depths")
                if isinstance(depths, list) and depths:
                    block["depths"] = [int(depths[0])]

        est_cfg = out.get("estimator")
        if isinstance(est_cfg, dict):
            est_cfg["label_noise_rates"] = [0.0]

        syn_cfg = out.get("synthetic_redirect")
        if not isinstance(syn_cfg, dict):
            syn_cfg = {}
        syn_cfg["enabled"] = False
        out["synthetic_redirect"] = syn_cfg

    return out


def _model_display_name(model: str) -> str:
    labels = {
        "instance_weighted": "instance_weighted",
        "aspic_v1_terminal": "aspic_v1_terminal",
        "aspic_v2": "aspic_v2",
        "aspic_v2_persistent": "aspic_v2_persistent",
        "schema_gating": "schema_gating",
    }
    return labels.get(model, model)


def _extract_sweep_row(
    dataset_run: Dict[str, object],
    condition: str = "clustered_redirect",
) -> Dict[str, Dict[str, float]]:
    conds = dataset_run.get("conditions", {})
    if not isinstance(conds, dict):
        conds = {}
    cond_row = conds.get(condition, {})
    if not isinstance(cond_row, dict):
        cond_row = {}
    out: Dict[str, Dict[str, float]] = {}
    for model in MODELS:
        row = cond_row.get(model, {})
        if not isinstance(row, dict):
            row = {}
        out[model] = {
            "chain_error_mean": float(row.get("chain_error_mean", 0.0)),
            "chain_error_std": float(row.get("chain_error_std", 0.0)),
            "cascade_mean": float(row.get("cascade_mean", 0.0)),
            "cascade_std": float(row.get("cascade_std", 0.0)),
            "accuracy_mean": float(row.get("accuracy_mean", 0.0)),
            "accuracy_std": float(row.get("accuracy_std", 0.0)),
        }
    return out


def run_corruption_rate_sweep(
    cfg: Dict[str, object],
    rates: List[float],
) -> Dict[str, object]:
    condition = "clustered_redirect"
    sweep: Dict[str, object] = {
        "condition": condition,
        "metric": "chain_error_mean",
        "rates": rates,
        "datasets": {},
    }

    has_ruletaker = isinstance(cfg.get("ruletaker"), dict)
    has_proofwriter = isinstance(cfg.get("proofwriter"), dict)
    datasets_out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    if has_ruletaker:
        datasets_out["ruletaker"] = {}
    if has_proofwriter:
        datasets_out["proofwriter"] = {}

    for rate in rates:
        cfg_rate = copy.deepcopy(cfg)
        noise_cfg = cfg_rate.get("noise")
        if not isinstance(noise_cfg, dict):
            noise_cfg = {}
        noise_cfg["clustered_rate"] = float(rate)
        cfg_rate["noise"] = noise_cfg
        key = f"{int(round(rate * 100))}%"

        if has_ruletaker:
            ruletaker = run_ruletaker_experiments(cfg_rate)
            datasets_out["ruletaker"][key] = _extract_sweep_row(ruletaker, condition=condition)
        if has_proofwriter:
            proofwriter = run_proofwriter_experiments(cfg_rate)
            datasets_out["proofwriter"][key] = _extract_sweep_row(proofwriter, condition=condition)

    sweep["datasets"] = datasets_out
    return sweep


def build_sweep_report(sweep: Dict[str, object]) -> str:
    lines: List[str] = []
    rates = sweep.get("rates", [])
    if not isinstance(rates, list):
        rates = []
    rate_labels = [f"{int(round(float(r) * 100))}%" for r in rates]
    datasets = sweep.get("datasets", {})
    if not isinstance(datasets, dict):
        datasets = {}

    lines.append("# Corruption Rate Sweep")
    lines.append("")
    lines.append(
        "Clustered redirect corruption sweep. Cells are chain-linked error "
        "(`chain_error_mean ± chain_error_std`)."
    )

    for dataset_key, by_rate in datasets.items():
        if not isinstance(by_rate, dict):
            continue
        lines.append("")
        lines.append(f"## {dataset_key.title()}")
        if not rate_labels:
            lines.append("- No rates provided.")
            continue
        lines.append("| Model | " + " | ".join(rate_labels) + " |")
        lines.append("|---|" + "---:|" * len(rate_labels))
        for model in MODELS:
            vals: List[str] = []
            for label in rate_labels:
                row = by_rate.get(label, {}).get(model, {})
                if not isinstance(row, dict):
                    row = {}
                vals.append(
                    f"{float(row.get('chain_error_mean', 0.0)):.3f} ± {float(row.get('chain_error_std', 0.0)):.3f}"
                )
            lines.append(f"| {_model_display_name(model)} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def _extract_tau_rows(
    dataset_run: Dict[str, object],
    conditions: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    conds = dataset_run.get("conditions", {})
    if not isinstance(conds, dict):
        conds = {}
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for condition in conditions:
        cond_row = conds.get(condition, {})
        if not isinstance(cond_row, dict):
            cond_row = {}
        out[condition] = {}
        for model in MODELS:
            row = cond_row.get(model, {})
            if not isinstance(row, dict):
                row = {}
            out[condition][model] = {
                "accuracy_mean": float(row.get("accuracy_mean", 0.0)),
                "accuracy_std": float(row.get("accuracy_std", 0.0)),
                "chain_error_mean": float(row.get("chain_error_mean", 0.0)),
                "chain_error_std": float(row.get("chain_error_std", 0.0)),
                "cascade_mean": float(row.get("cascade_mean", 0.0)),
                "cascade_std": float(row.get("cascade_std", 0.0)),
            }
    return out


def run_tau_sweep(
    cfg: Dict[str, object],
    taus: List[float],
) -> Dict[str, object]:
    focus_conditions = ["clustered_negate", "clustered_redirect"]
    sweep: Dict[str, object] = {
        "taus": taus,
        "conditions": focus_conditions,
        "datasets": {},
    }
    has_ruletaker = isinstance(cfg.get("ruletaker"), dict)
    has_proofwriter = isinstance(cfg.get("proofwriter"), dict)
    out: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    if has_ruletaker:
        out["ruletaker"] = {}
    if has_proofwriter:
        out["proofwriter"] = {}

    for tau in taus:
        cfg_tau = copy.deepcopy(cfg)
        est_cfg = cfg_tau.get("estimator")
        if not isinstance(est_cfg, dict):
            est_cfg = {}
        est_cfg["threshold"] = float(tau)
        cfg_tau["estimator"] = est_cfg
        key = f"{tau:.2f}"

        if has_ruletaker:
            rt = run_ruletaker_experiments(cfg_tau)
            out["ruletaker"][key] = _extract_tau_rows(rt, focus_conditions)
        if has_proofwriter:
            pw = run_proofwriter_experiments(cfg_tau)
            out["proofwriter"][key] = _extract_tau_rows(pw, focus_conditions)

    sweep["datasets"] = out
    return sweep


def build_tau_sweep_report(sweep: Dict[str, object]) -> str:
    lines: List[str] = []
    taus = sweep.get("taus", [])
    if not isinstance(taus, list):
        taus = []
    tau_labels = [f"{float(t):.2f}" for t in taus]
    conditions = sweep.get("conditions", [])
    if not isinstance(conditions, list):
        conditions = []
    datasets = sweep.get("datasets", {})
    if not isinstance(datasets, dict):
        datasets = {}

    lines.append("# Tau Sweep")
    lines.append("")
    lines.append(
        "Estimator threshold sweep. Cells are chain-linked error "
        "(`chain_error_mean ± chain_error_std`)."
    )

    for dataset_key, by_tau in datasets.items():
        if not isinstance(by_tau, dict):
            continue
        lines.append("")
        lines.append(f"## {dataset_key.title()}")
        for condition in conditions:
            lines.append("")
            lines.append(f"### {condition}")
            if not tau_labels:
                lines.append("- No tau values provided.")
                continue
            lines.append("| Model | " + " | ".join(f"τ={t}" for t in tau_labels) + " |")
            lines.append("|---|" + "---:|" * len(tau_labels))
            for model in MODELS:
                vals: List[str] = []
                for t in tau_labels:
                    row = (
                        by_tau.get(t, {})
                        .get(condition, {})
                        .get(model, {})
                    )
                    if not isinstance(row, dict):
                        row = {}
                    vals.append(
                        f"{float(row.get('chain_error_mean', 0.0)):.3f} ± {float(row.get('chain_error_std', 0.0)):.3f}"
                    )
                lines.append(f"| {_model_display_name(model)} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    configure_hf_cache_root()
    cfg_path = Path(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg = apply_cli_overrides(cfg, args)

    hardware = detect_hardware()
    save_json(outdir / "hardware.json", hardware)

    sweep_rates = parse_sweep_rates(args.sweep_rates) if args.sweep_rates.strip() else []
    tau_values = parse_tau_values(args.tau_sweep) if args.tau_sweep.strip() else []
    _assert_paper_locked_args(args, sweep_rates, tau_values)
    if sweep_rates and tau_values:
        raise ValueError("Use either --sweep-rates or --tau-sweep in a single run, not both.")
    if sweep_rates:
        print(f"Running clustered redirect sweep for rates: {', '.join(str(r) for r in sweep_rates)}")
        sweep = run_corruption_rate_sweep(cfg, sweep_rates)
        payload = {
            "hardware": hardware,
            "config": cfg,
            "sweep": sweep,
        }
        save_json(outdir / "sweep_results.json", payload)
        with open(outdir / "sweep_report.md", "w", encoding="utf-8") as handle:
            handle.write(build_sweep_report(sweep) + "\n")
        print(f"Saved sweep outputs to: {outdir}")
        return 0
    if tau_values:
        print(f"Running tau sweep for values: {', '.join(f'{t:.2f}' for t in tau_values)}")
        tau_sweep = run_tau_sweep(cfg, tau_values)
        payload = {
            "hardware": hardware,
            "config": cfg,
            "tau_sweep": tau_sweep,
        }
        save_json(outdir / "tau_sweep_results.json", payload)
        with open(outdir / "tau_sweep_report.md", "w", encoding="utf-8") as handle:
            handle.write(build_tau_sweep_report(tau_sweep) + "\n")
        print(f"Saved tau sweep outputs to: {outdir}")
        return 0

    if args.diagnose_matching:
        diag_cfg = copy.deepcopy(cfg)
        if args.dataset_key == "proofwriter":
            pw_cfg = cfg.get("proofwriter", {})
            if isinstance(pw_cfg, dict):
                reason_cfg = diag_cfg.get("reasoner")
                if not isinstance(reason_cfg, dict):
                    reason_cfg = {}
                reason_cfg["default_policy"] = str(pw_cfg.get("default_policy", "random_tie"))
                diag_cfg["reasoner"] = reason_cfg
        base = load_problems_base(diag_cfg, dataset_key=args.dataset_key)
        first_seed = int(cfg["seeds"][0]) if cfg.get("seeds") else 42
        clean = apply_noise(base, "clean", first_seed, diag_cfg)
        diag = build_matching_diagnostics(clean, sample_count=max(1, args.diagnose_samples))
        save_json(outdir / "matching_diagnostic.json", diag)
        payload = {
            "hardware": hardware,
            "dataset_key": args.dataset_key,
            "status": "matching_diagnostic_completed",
            "diagnostic_summary": {
                "sample_count": diag["sample_count"],
                "antecedent_match_rate_in_initial_facts": diag[
                    "antecedent_match_rate_in_initial_facts"
                ],
                "question_in_initial_facts_rate": diag["question_in_initial_facts_rate"],
                "question_in_any_schema_consequent_rate": diag[
                    "question_in_any_schema_consequent_rate"
                ],
            },
        }
        save_json(outdir / "results.json", payload)
        print(f"Saved matching diagnostics to: {outdir}")
        return 0

    ruletaker: Optional[Dict[str, object]] = None
    if isinstance(cfg.get("ruletaker"), dict):
        ruletaker = run_ruletaker_experiments(cfg)
    proofwriter: Optional[Dict[str, object]] = None
    if isinstance(cfg.get("proofwriter"), dict):
        print("Running ProofWriter experiments...")
        proofwriter = run_proofwriter_experiments(cfg)
    if ruletaker is None and proofwriter is None:
        raise ValueError("Config must define at least one dataset block: `ruletaker` or `proofwriter`.")

    # Save parsed/noisy dataset metadata for review and reproducibility.
    dataset_meta: Dict[str, object] = {}
    dataset_runs: Dict[str, Dict[str, object]] = {}
    if ruletaker is not None:
        dataset_runs["ruletaker"] = ruletaker
    if proofwriter is not None:
        dataset_runs["proofwriter"] = proofwriter

    for dataset_name in list(dataset_runs.keys()):
        dataset_runs[dataset_name]["connectivity_analysis"] = compute_dataset_connectivity_analysis(
            dataset_runs[dataset_name]
        )

    for dataset_name, dataset_results in dataset_runs.items():
        per_dataset_meta: Dict[str, object] = {}
        for cond, cond_map in dataset_results["datasets"].items():
            per_seed: Dict[str, object] = {}
            cond_flags: Dict[str, List[str]] = {}
            for seed, probs in cond_map.items():
                summary = summarize_problems(probs)
                flags = quality_flags_for_summary(summary)
                summary["quality_flags"] = flags
                per_seed[str(seed)] = summary
                cond_flags[str(seed)] = flags

            per_dataset_meta[cond] = {
                "seeds": per_seed,
                "seed_quality_flags": cond_flags,
            }
        dataset_meta[dataset_name] = per_dataset_meta
    save_json(outdir / "dataset_summary.json", dataset_meta)

    if args.prepare_only:
        payload = {
            "hardware": hardware,
            "dataset_summary": dataset_meta,
            "status": "prepare_only_completed",
        }
        save_json(outdir / "results.json", payload)
        print(f"Prepared datasets and metadata at: {outdir}")
        return 0

    skip_ablation = bool(args.skip_ablation or args.smoke_test)
    skip_synthetic = bool(args.skip_synthetic or args.smoke_test)

    ablation: Dict[str, Dict[str, Dict[str, float]]] = {}
    ablation_dataset_key: Optional[str] = None
    if not skip_ablation and ruletaker is not None:
        ablation = run_degraded_ablation(cfg, ruletaker["datasets"])
        ablation_dataset_key = "ruletaker"
    elif not skip_ablation and proofwriter is not None:
        ablation = run_degraded_ablation(cfg, proofwriter["datasets"])
        ablation_dataset_key = "proofwriter"
    synthetic_cfg = cfg.get("synthetic_redirect", {})
    if not isinstance(synthetic_cfg, dict):
        synthetic_cfg = {}
    run_synthetic = bool(synthetic_cfg.get("enabled", True))
    n_synthetic = int(synthetic_cfg.get("n_problems", 60))
    synthetic_redirect: Dict[str, object] = {}
    if run_synthetic and not skip_synthetic:
        print("Running synthetic redirect experiment...")
        synthetic_redirect = run_synthetic_redirect_experiment(cfg, n_problems=n_synthetic)

    # Drop in-memory datasets before serializing results.
    ruletaker_no_data: Optional[Dict[str, object]] = None
    if ruletaker is not None:
        ruletaker_no_data = {
            k: v for k, v in ruletaker.items() if k != "datasets"
        }
    proofwriter_no_data: Optional[Dict[str, object]] = None
    if proofwriter is not None:
        proofwriter_no_data = {k: v for k, v in proofwriter.items() if k != "datasets"}
    results = {
        "hardware": hardware,
        "ablation": ablation,
        "ablation_dataset_key": ablation_dataset_key,
        "synthetic_redirect": synthetic_redirect,
        "config": cfg,
    }
    if ruletaker_no_data is not None:
        results["ruletaker"] = ruletaker_no_data
    if proofwriter_no_data is not None:
        results["proofwriter"] = proofwriter_no_data
    connectivity_analysis: Dict[str, object] = {}
    if ruletaker_no_data is not None and isinstance(ruletaker_no_data.get("connectivity_analysis"), dict):
        connectivity_analysis["ruletaker"] = ruletaker_no_data["connectivity_analysis"]
    if proofwriter_no_data is not None and isinstance(proofwriter_no_data.get("connectivity_analysis"), dict):
        connectivity_analysis["proofwriter"] = proofwriter_no_data["connectivity_analysis"]
    results["connectivity_analysis"] = connectivity_analysis
    save_json(outdir / "results.json", results)

    report = build_report(
        hardware=hardware,
        ruletaker=ruletaker_no_data or {},
        ablation=ablation,
        cfg=cfg,
        synthetic_redirect=synthetic_redirect,
        proofwriter=proofwriter_no_data,
        ablation_dataset_key=ablation_dataset_key,
    )
    with open(outdir / "report.md", "w", encoding="utf-8") as handle:
        handle.write(report + "\n")

    print(f"Saved outputs to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
