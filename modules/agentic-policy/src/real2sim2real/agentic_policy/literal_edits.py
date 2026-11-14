"""Narrow author edits without permitting changes to control flow or guards."""

import ast
import math


def bounded_keyword_vectors(seed, candidate, *, keyword, bounds, require_same=True):
    """Allow only constant vector literals at existing named-call keyword slots.

    All source bytes outside those vectors must remain unchanged, including
    guards, timing and comments. No evaluation or import of candidate code occurs.
    Bounds are inclusive per component. This contract complements, and never
    replaces, sandboxing, robot joint limits and live simulation checks.
    """
    if not bounds or any(
        len(pair) != 2 or not all(math.isfinite(x) for x in pair) or pair[0] > pair[1]
        for pair in bounds
    ):
        raise ValueError("Finite ordered per-component bounds required")

    def inspect(source):
        tree = ast.parse(source)
        lines = source.encode("utf-8").splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        spans, vectors = [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for item in node.keywords:
                if item.arg != keyword:
                    continue
                value = item.value
                if not isinstance(value, ast.List):
                    raise TypeError("Only literal lists may occupy permitted slots")
                vector = ast.literal_eval(value)
                if len(vector) != len(bounds) or any(
                    type(x) not in (int, float)
                    or not math.isfinite(x)
                    or not low <= x <= high
                    for x, (low, high) in zip(vector, bounds)
                ):
                    raise ValueError("Vector outside declared bounds")
                spans.append(
                    (
                        offsets[value.lineno - 1] + value.col_offset,
                        offsets[value.end_lineno - 1] + value.end_col_offset,
                    )
                )
                vectors.append(vector)
        if not spans:
            raise ValueError("No permitted vector slots in source")
        data = source.encode("utf-8")
        for start, end in sorted(spans, reverse=True):
            data = data[:start] + b"<bounded-vector>" + data[end:]
        if require_same and any(v != vectors[0] for v in vectors):
            raise ValueError("All permitted vectors must be identical")
        return data, vectors

    masked_seed, original = inspect(seed)
    masked_candidate, vectors = inspect(candidate)
    if masked_seed != masked_candidate or len(original) != len(vectors):
        raise ValueError("Candidate changed code outside permitted vector literals")
    return {
        "passed": True,
        "keyword": keyword,
        "slots": len(vectors),
        "vectors": vectors,
    }
