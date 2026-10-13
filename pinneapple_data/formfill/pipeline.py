"""One call from source documents to a compiled form, and the helpers a front end needs around it.

    from pinneapple_data.formfill import compile_documents, get_spec
    comp, rejected = compile_documents([("rfq.pdf", data, "customer")], get_spec("api610_pump"))
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .extract import Compilation, check_answer, compile, extract_rules, read_document
from .llm import extract_with_llm
from .spec import FormSpec


def compile_documents(docs: Sequence[Tuple[str, bytes, str]], spec: FormSpec, client: Any = None,
                      overrides: Optional[Mapping[str, Any]] = None,
                      max_ocr_pages: Optional[int] = None) -> Tuple[Compilation, List[Dict[str, Any]]]:
    """Read every ``(name, data, role)`` document, extract candidates with the rules of ``spec`` (and with the local
    LLM ``client`` too, when given) and compile them into one decision per form item.

    A failing LLM extraction never fails the call: the rule-based candidates stand and the failure is reported.
    Returns ``(compilation, rejected)``; ``rejected`` lists the LLM candidates that were dropped, each with its
    ``reason`` and ``doc``.
    """
    read_kw = {} if max_ocr_pages is None else {"max_ocr_pages": max_ocr_pages}
    read, cands, rejected = [], [], []
    for name, data, role in docs:
        d = read_document(data, name=name, role=role, **read_kw)
        read.append(d)
        cands.extend(extract_rules(d, spec))
        if client is not None:
            try:
                acc, rej = extract_with_llm(d, spec, client)
            except Exception as exc:  # noqa: BLE001 - the rule-based result still stands
                acc, rej = [], [{"reason": f"local LLM extraction failed: {type(exc).__name__}: {exc}"}]
            cands.extend(acc)
            rejected.extend({**r, "doc": name} for r in rej)
    return compile(read, cands, spec, overrides=dict(overrides or {})), rejected


def check_overrides(spec: FormSpec, raw: Mapping[str, Any]) -> Dict[str, Any]:
    """User answers ``{item key: value}`` checked against ``spec`` (:func:`check_answer`): empty values are dropped,
    an unknown key or an invalid value raises ``ValueError`` with a message for the user."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k not in spec.by_key:
            raise ValueError(f"unknown item '{k}'")
        if v in (None, ""):
            continue
        out[k] = check_answer(spec[k], v)
    return out


def output_filename(comp: Compilation, prefix: str = "", day: Optional[date] = None) -> str:
    """File name for the filled form: ``<spec id>_<value of spec.filename_key or "filled">_<YYYYMMDD>.pdf``, made of
    safe characters only."""
    spec = comp.spec
    tag = comp.decisions[spec.filename_key].display if spec.filename_key in comp.decisions else ""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{spec.id}_{tag or 'filled'}")
    return f"{prefix}{base}_{(day or date.today()):%Y%m%d}.pdf"


__all__ = ["compile_documents", "check_overrides", "output_filename"]
