#!/usr/bin/env python3
"""Parity smoke test for Math-by-Design browser preview vs canonical Python palette engine.

The browser compiler is intentionally a heuristic preview today. This test ensures a
known mismatch cannot silently masquerade as canonical parity: compiler output must
either equal the Python engine or the master page must explicitly declare the preview
as heuristic and parity as not asserted.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASTER = HERE / "math-by-design.html"
ENGINE = HERE / "fibonacci_hex_palettes.py"
SEED = "#FF6B45"
FAMILY = "fibonacci"


def extract_function(js: str, name: str) -> str:
    start = js.index(f"function {name}(")
    brace = js.index("{", start)
    depth = 0
    quote = None
    esc = False
    for i in range(brace, len(js)):
        ch = js[i]
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"`":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return js[start:i + 1]
    raise ValueError(f"unterminated function {name}")


def extract_const(js: str, name: str) -> str:
    m = re.search(rf"const {re.escape(name)}=(.*?);\n", js, re.S)
    if not m:
        raise ValueError(f"const {name} not found")
    return f"const {name}={m.group(1)};"


def browser_nodes(master_text: str) -> list[str]:
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", master_text, re.S)
    js = scripts[-1]
    pieces = [
        extract_const(js, "PHI"),
        extract_const(js, "TRIB"),
        extract_const(js, "SILVER"),
        extract_const(js, "PLASTIC"),
        extract_const(js, "HALF_PI"),
        extract_const(js, "SPIRALS"),
    ]
    for fn in ["clamp", "hexToRgb", "rgbToHex", "mix", "rgbToHsl", "hslToRgb", "familyRatio", "derivePalette"]:
        pieces.append(extract_function(js, fn))
    pieces.append(f"console.log(JSON.stringify(derivePalette('{SEED}','{FAMILY}','nodes','#0B0F14')));")
    proc = subprocess.run(["node", "-e", "\n".join(pieces)], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def engine_nodes() -> list[str]:
    sys.path.insert(0, str(HERE))
    import fibonacci_hex_palettes as engine
    _, palettes = engine.generate_palettes(
        SEED, families=[FAMILY], direction="in", rauzy=False, geometry=False, baselines=False
    )
    return palettes[0].hexes


def main() -> int:
    master = MASTER.read_text(encoding="utf-8")
    preview = browser_nodes(master)
    canonical = engine_nodes()
    equal = preview == canonical
    explicitly_flagged = (
        "tokens_source:'heuristic-preview'" in master
        and "parity_status:'not asserted;" in master
        and "canonical_palette_engine:SOURCES.paletteEngine" in master
    )
    seed_exact = bool(preview) and preview[0].upper() == SEED
    result = {
        "seed": SEED,
        "family": FAMILY,
        "browser_preview": preview,
        "canonical_engine": canonical,
        "equal": equal,
        "explicitly_flagged_if_not_equal": explicitly_flagged,
        "seed_exact_in_preview": seed_exact,
        "pass": seed_exact and (equal or explicitly_flagged),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
