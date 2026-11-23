# Explicit Complete Intersection Generators for the Integral Hodge Lattice of the Fermat Sextic Fourfold and Surface

I use AI for creating documents and organizing scripts.

This repository contains the manuscript and deterministic, self-contained computational verification scripts accompanying:

> **Explicit Complete Intersection Generators for the Integral Hodge Lattice of the Fermat Sextic Fourfold and Surface**

## Directory Layout

```text
explicit-hodge-fermat-sextic/
├── README.md                              # This file
├── paper/
│   ├── paper_en.tex                       # English manuscript (LaTeX source)
│   ├── paper_en.pdf                       # Compiled PDF (10 pages, zero warnings)
│   └── paper_ja.md                        # Japanese summary and comparison table
└── verification/
    ├── verify_surface_x62.sage            # Theorem A, Prop 3.1(1), and Table 1 (~5 s)
    ├── verify_5_surfaces_geometry.sage    # Prop 3.1(2): CI dimension & isolated singularities (~10 s)
    ├── sextic_padic_exact.py              # Theorem B: exact p-adic SNFs & 2-layer gluing (~6 min)
    ├── export_cycle_templates.sage        # Exact Griffiths residue exporter (P, Z_J11^Aoki, ...)
    ├── export_c4_template.sage            # Exact Griffiths residue exporter (Z_DEC20^C4)
    ├── export_cube_b6_templates.sage      # Exact Griffiths residue exporter (Z_DEC20^cube, Z_J11^cube)
    ├── referee_certify_exact.sage         # Zero-float FLINT exact det(G_B0)=2^2799*3^1314 + J11 v3=30 (~6 min)
    ├── test_c4_on_x64.py                  # Theorem C(1): Z[1/2]-IHC, index 2^426 (~2 min)
    ├── test_full_z_ihc_x64.py             # Theorem C(2): full Z-IHC, index 1 (~4 min)
    ├── diagnose_x64_cokernel.py           # Layer-by-layer cokernel: v2(DEC20)=380, v2(J11)=46
    ├── audit_against_rafaty.py            # Section 6.3(1): Pham bridge & sign audit of Z_DEC20^C4
    ├── audit_truth_serum.py               # Section 6.3(2-3): direct K3.npy 70/70 mod 3 cross-check (~24 s)
    └── data/
        ├── cycle_templates_mod_q.json
        ├── cycle_templates_mod_q_v2.json
        ├── cycle_templates_mod_q_v3.json
        ├── sextic_full_lattice_results.json
        ├── x64_ihc_c4_results.json
        └── x64_full_z_ihc_results.json
```

## Software Requirements

- **SageMath 10.x** (tested on SageMath 10.10 with FLINT and Singular)
- **Python 3.11+ with NumPy 2.x** (bundled with SageMath)
- **TeX Live** (`pdflatex` with standard `amsart`, `amsmath`, `amssymb`, `amsthm`, `array`, `hyperref`) to rebuild `paper/paper_en.pdf`

## Quick Verification Commands

Run all commands from the `verification/` directory:

```bash
cd verification

# 1. Theorem A, Proposition 3.1(1), and all 4 rows of Table 1 (exact over Z, ~5 s)
sage verify_surface_x62.sage

# 2. Proposition 3.1(2): complete intersection dimension & isolated singularities on X_6^4 (~10 s)
sage verify_5_surfaces_geometry.sage

# 3. Theorem B: exact 2-adic and 3-adic SNFs & two-layer gluing of all Shioda strata (~6 min)
python sextic_padic_exact.py

# 4. Theorem C (zero-float FLINT 1469-digit exact det(G_B0) = 2^2799 * 3^1314, ~6 min)
sage referee_certify_exact.sage

# 5. Theorem C(1): Z[1/2]-IHC ([S : <P, Z_DEC20^C4, Z_J11^Aoki>] = 2^426, ~2 min)
python test_c4_on_x64.py

# 6. Theorem C(2): Full Z-IHC ([Hdg^2(X_6^4, Z) : <5 families>] = 1, ~4 min)
python test_full_z_ihc_x64.py

# 7. (Optional) Regenerate all dual split-prime (q1=10009, q2=10333) templates from scratch
sage export_cycle_templates.sage
sage export_c4_template.sage
sage export_cube_b6_templates.sage

# 8. Section 6.3 cross-verification against Santana de Jesus (2026)
python audit_against_rafaty.py
# To run audit_truth_serum.py directly on Santana de Jesus's published K3.npy.xz:
# git clone https://github.com/Rafaty/integral-hodge-fermat-sextic external/integral-hodge-fermat-sextic
python audit_truth_serum.py external/integral-hodge-fermat-sextic/verifier/data/g13/K3.npy.xz
```

## Notes on Priority & Cross-Verification

- The first proof of the Integral Hodge Conjecture for $X_6^4$ is due to **Rafael Santana de Jesus** (Nov 1, 2026; [github.com/Rafaty/integral-hodge-fermat-sextic](https://github.com/Rafaty/integral-hodge-fermat-sextic)).
- The $\mathbb{Z}[\frac{1}{6}]$-basis of surfaces of types $(1,1,1), (1,2,3), (2,2,3)$ is due to **Braun–Fortin–Lopez Garcia–Villaflor Loyola** (*JHEP* 2024).
- Our main contribution on $X_6^4$ is resolving **Santana de Jesus Remark 8(1)** by providing a completely explicit equation-defined generating set of just five normal reduced complete intersection families over $K = \mathbb{Q}(\zeta_{12}, \sqrt[3]{2})$ (eliminating all non-explicit Lefschetz $(1,1)$ correspondences and blow-up transports), together with Theorem A (minimal 3-family integral generation of $\mathrm{NS}(X_6^2)$) and Theorem B (complete Shioda-stratum SNFs and two-layer $3$-adic gluing).
