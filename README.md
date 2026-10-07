# Math by Design

A small agent-facing design grammar for building interfaces from **hierarchy, material, mathematical color relationships, and interaction behavior** without collapsing those decisions into one reusable template.

The project is intentionally split into human/dev references and a callable palette engine:

- **`math-by-design.html`** — master router/compiler/playground. Start here.
- **`fibonacci-landing-page-template.html`** — canonical Phi Flow `8 → 5 → 3 → 2 → 1 → 1` layout contract.
- **`phi-flow-dev-ref.html`** — hierarchy handoffs, dev inspection, drift checks, multi-agent protocol, F1–F5 gate.
- **`ui-build-ref-master-v2.html`** — application-to-style routing, material languages, anti-template rules, techniques and build order.
- **`fibonacci_hex_convergence_lab.html`** — original single-family Fibonacci HEX convergence lab.
- **`fibonacci_hex_multispiral_rauzy_lab.html`** — multi-family spiral color generation plus Tribonacci Rauzy A/B/C spatial sampling.
- **`fibonacci_hex_multispiral_rauzy_geometry_lab.html`** — interactive palette geometry analysis in the RGB cube with PCA-style shape metrics, hull/gray-axis measures and null-model context.
- **`fibonacci_hex_palettes.py`** — canonical reproducible SymPy/NumPy palette engine for agents and build tooling.

## The separation that matters

`Phi Flow` answers **where attention moves**.  
Material language answers **what the interface feels made of**.  
The math/color layer answers **how related colors are generated**.  
Interaction rules answer **how temporary UI hands attention to semantic content**.

These are references, not templates. Do not clone an entire palette + type + geometry + effect signature from any example.

## Python palette engine

Requires Python with `numpy` and `sympy`. `scipy` is optional and enables 3-D convex-hull volume.

```bash
python fibonacci_hex_palettes.py --self-test
python fibonacci_hex_palettes.py "#5CF2B2"
python fibonacci_hex_palettes.py "#5CF2B2" --direction both --count 8
python fibonacci_hex_palettes.py "#C0392B" \
  --families fibonacci tribonacci plastic binet \
  --direction both \
  --count 8 \
  --rauzy-spin 4 \
  --json palette-candidates.json \
  --html palette-candidates.html
```

The geometry report is descriptive rather than a beauty score. It can classify a palette as a line/spine, triangle, planar wedge, folded fan, or volume/cloud and compare flatness against null models; accessibility still has to be validated from the rendered interface.

## Suggested agent workflow

1. Read `math-by-design.html` first.
2. Assign semantic flow before styling.
3. Choose material language independently from geometry.
4. Use `fibonacci_hex_palettes.py` only when reproducible generated color relationships or palette-geometry provenance are useful.
5. Clamp/alter generated colors as needed for actual UI contrast and record the change.
6. Run the F1–F5 validation gate before expanding visual effects.
7. Preserve design lineage with exact source names and SHA-256s.

## License

See [`LICENSE`](LICENSE).
