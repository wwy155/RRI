"""Render-Reading Instruments: read a registered item with its instrument and
commit the reading when the validity gate passes; every other item is
deferred to the MLLM."""
import os
import re

from . import counter, transient


def _read_transient(item, paths):
    roles = {}
    for p in paths:
        for role in ("science", "reference", "difference"):
            if "_%s_" % role in os.path.basename(p):
                roles[role] = transient.load_gray(p)
    if len(roles) < 3:
        return None, {}
    verdict, sources = transient.compare(roles["science"], roles["reference"],
                                         roles["difference"])
    return transient.bind_option(verdict, item.get("options")), {"verdict": verdict,
                                                                 "sources": sources}


def _read_counter(item, paths):
    b, y, valid = counter.count_labels(paths[0])
    keys = re.findall(r'"([^"]+)"\s*:', str(item.get("question") or ""))  # answer template
    kb = next((k for k in keys if k.lower().startswith("b")), "b ions")
    ky = next((k for k in keys if k.lower().startswith("y")), "y ions")
    answer = '{\n  "%s": %d,\n  "%s": %d\n}' % (kb, b, ky, y) if valid else None
    return answer, {"b_ions": b, "y_ions": y}


REGISTRY = {"A011": _read_transient,   # transient detection (astronomy)
            "L001": _read_counter}     # peptide-fragment counting (life science)


def read(item, paths):
    """Reading of a registered item: {'answer': committed answer or None, 'reading': ...}.

    Returns None for tasks without an instrument.  The reference answer is never read.
    """
    fn = REGISTRY.get(str(item["id"]).split("/")[0])
    if fn is None:
        return None
    answer, reading = fn(item, paths)
    return {"answer": answer, "reading": reading}
