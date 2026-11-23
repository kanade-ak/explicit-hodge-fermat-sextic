"""
Exact p-adic local-to-global verification of Theorem B for X_6^4:
  - Proves N_{Q(zeta_6)/Q}(det F) = 2^6 * 3^6 (prime support {2, 3} only).
  - Computes p-adic saturated kernels, Gram matrices, and SNFs at p=2 (K=26) and p=3 (K=17)
    for L, S, T, V, W, W_DEC20, W_J11 and their orthogonal gluing groups.
  - Verifies the three gluing determinant identities and saves data/sextic_full_lattice_results.json.
"""

import itertools
import json
import math
import os
import time
from collections import Counter
import numpy as np

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.2f}s] {msg}", flush=True)


def prove_prime_support_2_3():
    # Z[zeta_6]: zeta^2 - zeta + 1 = 0
    z_pow = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def zadd(x, y):
        return (x[0] + y[0], x[1] + y[1])

    def zsub(x, y):
        return (x[0] - y[0], x[1] - y[1])

    def zmul(x, y):
        return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0] + x[1] * y[1])

    F = [[zsub(z_pow[((b + 1) * (bp + 1)) % 6], z_pow[(b * (bp + 1)) % 6]) for bp in range(5)] for b in range(5)]

    detF = (0, 0)
    for perm in itertools.permutations(range(5)):
        inv = sum(1 for i in range(5) for j in range(i + 1, 5) if perm[i] > perm[j])
        term = (1, 0) if (inv % 2 == 0) else (-1, 0)
        for i in range(5):
            term = zmul(term, F[i][perm[i]])
        detF = zadd(detF, term)

    a, b = detF
    norm_detF = a * a + a * b + b * b
    tmp = norm_detF
    v2 = 0
    while tmp % 2 == 0:
        tmp //= 2
        v2 += 1
    v3 = 0
    while tmp % 3 == 0:
        tmp //= 3
        v3 += 1
    assert tmp == 1, f"Unexpected prime factor in N(det F): remainder={tmp}"
    log(f"5x5 period matrix det(F) = {a} + ({b})*zeta_6, Norm = 2^{v2} * 3^{v3} = {norm_detF}")
    return {"detF": [a, b], "norm_v2": v2, "norm_v3": v3}


def build_pham_and_slice():
    r = 5
    I = list(itertools.product(range(r), repeat=5))
    mu = len(I)
    assert mu == 3125
    idx_I = {beta: i for i, beta in enumerate(I)}

    Psi = np.zeros((mu, mu), dtype=np.int64)
    deltas = [dt for dt in itertools.product((0, 1), repeat=5) if any(dt)]
    delta_sgns = [(-1 if (sum(dt) % 2 == 1) else 1) for dt in deltas]
    for i, beta in enumerate(I):
        Psi[i, i] = 2
        b0, b1, b2, b3, b4 = beta
        for (d0, d1, d2, d3, d4), sgn in zip(deltas, delta_sgns):
            c0, c1, c2, c3, c4 = b0 + d0, b1 + d1, b2 + d2, b3 + d3, b4 + d4
            if c0 < r and c1 < r and c2 < r and c3 < r and c4 < r:
                j = idx_I[(c0, c1, c2, c3, c4)]
                Psi[i, j] = sgn
                Psi[j, i] = sgn

    # Extract 2605 deterministic pivot columns over GF(1009)
    p = 1009
    A = (Psi % p).copy()
    pivs = []
    row = 0
    for col in range(mu):
        nz = np.nonzero(A[row:, col])[0]
        if nz.size == 0:
            continue
        piv_r = row + int(nz[0])
        if piv_r != row:
            A[[row, piv_r]] = A[[piv_r, row]]
        inv = pow(int(A[row, col]), -1, p)
        A[row, col:] = (A[row, col:] * inv) % p
        if row + 1 < mu:
            below = A[row + 1:, col]
            nzb = np.nonzero(below)[0]
            if nzb.size > 0:
                ri = row + 1 + nzb
                A[ri, col:] = (A[ri, col:] - np.outer(below[nzb], A[row, col:])) % p
        pivs.append(col)
        row += 1
        if row == mu:
            break

    assert len(pivs) == 2605, len(pivs)
    Psi_L = Psi[np.ix_(pivs, pivs)].copy()
    I_L = [I[idx] for idx in pivs]
    log("Built Psi (3125x3125) and extracted rank-2605 slice over GF(1009)")
    return I, I_L, Psi_L


def classify_forms_and_build_A2(I, I_L):
    I1, I_LIN, I_DEC20, I_J11 = [], [], [], []
    for beta in I:
        s = sum(b + 1 for b in beta)
        if s < 12 and (s % 6 != 0):
            I1.append(beta)
        b0 = 18 - s - 1
        if 0 <= b0 <= 4:
            full_a = (b0 + 1,) + tuple(b + 1 for b in beta)
            c = Counter(full_a)
            is_lin = all((cnt % 2 == 0 if x == 3 else c.get(6 - x, 0) == cnt) for x, cnt in c.items())
            if is_lin:
                I_LIN.append(beta)
            elif any((6 - x) in c and (x != 3 or c[x] >= 2) for x in c):
                I_DEC20.append(beta)
            else:
                I_J11.append(beta)

    assert (len(I1), len(I_LIN), len(I_DEC20), len(I_J11)) == (427, 1001, 720, 30)

    def half_conj(forms):
        seen = set()
        out = []
        for bp in forms:
            key = min(bp, tuple(4 - x for x in bp))
            if key not in seen:
                seen.add(key)
                out.append(bp)
        return out

    I_DEC20_h = half_conj(I_DEC20)
    I_J11_h = half_conj(I_J11)
    I_LIN_h = half_conj(I_LIN)
    I2_h = I_DEC20_h + I_J11_h

    z_pow = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    fac = [
        [
            (
                z_pow[((b + 1) * (bp + 1)) % 6][0] - z_pow[(b * (bp + 1)) % 6][0],
                z_pow[((b + 1) * (bp + 1)) % 6][1] - z_pow[(b * (bp + 1)) % 6][1],
            )
            for bp in range(5)
        ]
        for b in range(5)
    ]

    def build_A2(forms_list):
        nrows = len(I_L)
        ncols = 2 * len(forms_list)
        M = np.empty((nrows, ncols), dtype=np.int64)
        for r_idx, beta in enumerate(I_L):
            f0, f1, f2, f3, f4 = fac[beta[0]], fac[beta[1]], fac[beta[2]], fac[beta[3]], fac[beta[4]]
            col = 0
            for bp in forms_list:
                ra, rb = f0[bp[0]]
                for fk, idx in ((f1, bp[1]), (f2, bp[2]), (f3, bp[3]), (f4, bp[4])):
                    ca, cb = fk[idx]
                    ra, rb = ra * ca - rb * cb, ra * cb + rb * ca + rb * cb
                M[r_idx, col] = ra
                M[r_idx, col + 1] = rb
                col += 2
        return M

    A2_I1 = build_A2(I1)
    A2_I2 = build_A2(I2_h)
    A2_LIN = build_A2(I_LIN_h)
    A2_DEC20 = build_A2(I_DEC20_h)
    A2_J11 = build_A2(I_J11_h)
    log(f"Built period matrices: A2(I1)={A2_I1.shape}, A2(I2_h)={A2_I2.shape}, A2(LIN_h)={A2_LIN.shape}, A2(DEC20_h)={A2_DEC20.shape}, A2(J11_h)={A2_J11.shape}")
    return A2_I1, A2_I2, A2_LIN, A2_DEC20, A2_J11


def matmul_mod(A, B, mod, blk=32):
    """Blocked (A @ B) % mod avoiding int64 overflow for mod <= 2^28."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    C = np.zeros((m, n), dtype=np.int64)
    A_m = A % mod
    B_m = B % mod
    for s in range(0, k, blk):
        e = min(k, s + blk)
        C = (C + A_m[:, s:e] @ B_m[s:e, :]) % mod
    return C


def padic_left_kernel(A_in, p, K, expected_rank=None, label=""):
    """Saturated Z_p-left kernel of A_in over Z/p^K Z; effective precision is K - m_piv."""
    t0 = time.time()
    N, M = A_in.shape
    mod = int(p ** K)
    p_powers = [int(p ** j) for j in range(K + 1)]
    B = (A_in % mod).copy()
    U = np.eye(N, dtype=np.int64)

    piv_vals = []
    max_steps = min(N, M)
    for step in range(max_steps):
        sub = B[step:, step:]
        best_v = K
        rel_i = -1
        rel_j = -1
        for v in range(K):
            mask = (sub % p_powers[v + 1]) != 0
            if np.any(mask):
                best_v = v
                flat = int(np.argmax(mask))
                rel_i, rel_j = divmod(flat, sub.shape[1])
                break
        if best_v == K:
            break

        piv_vals.append(best_v)
        bi = step + rel_i
        bj = step + rel_j
        if bi != step:
            B[[step, bi]] = B[[bi, step]]
            U[[step, bi]] = U[[bi, step]]
        if bj != step:
            B[:, [step, bj]] = B[:, [bj, step]]

        pv = p_powers[best_v]
        mod_red = mod // pv
        piv_unit = int(B[step, step] // pv) % mod_red
        inv_unit = pow(piv_unit, -1, mod_red)

        if step + 1 < N:
            col_below = B[step + 1:, step]
            nz = np.nonzero(col_below)[0]
            if nz.size > 0:
                rows_idx = step + 1 + nz
                mults = ((col_below[nz] // pv) * inv_unit) % mod_red
                B[rows_idx, step:] = (B[rows_idx, step:] - np.outer(mults, B[step, step:])) % mod
                U[rows_idx, :] = (U[rows_idx, :] - np.outer(mults, U[step, :])) % mod

    rk = len(piv_vals)
    if expected_rank is not None:
        assert rk == expected_rank, f"{label} (p={p}): rank={rk} != expected {expected_rank}"
    if rk < N:
        assert not np.any(B[rk:, :]), f"{label} (p={p}): residual non-zero!"
    m_piv = max(piv_vals) if piv_vals else 0
    U_ker = U[rk:, :].copy()
    log(f"    [ker {label}, p={p}] {N}x{M} -> rk={rk}, ker={N-rk}, max_piv_val={m_piv} (K_eff={K-m_piv}, {time.time()-t0:.2f}s)")
    return U_ker, m_piv


def padic_snf_vals(M_in, p, K_eff, label=""):
    """Diagonal p-valuations of SNF(M_in) over Z/p^{K_eff}Z; asserts max(vals) < K_eff."""
    t0 = time.time()
    r = M_in.shape[0]
    assert M_in.shape == (r, r)
    mod = int(p ** K_eff)
    p_powers = [int(p ** j) for j in range(K_eff + 1)]
    A = (M_in % mod).copy()
    vals = []

    for step in range(r):
        sub = A[step:, step:]
        best_v = K_eff
        rel_i = -1
        rel_j = -1
        for v in range(K_eff):
            mask = (sub % p_powers[v + 1]) != 0
            if np.any(mask):
                best_v = v
                flat = int(np.argmax(mask))
                rel_i, rel_j = divmod(flat, sub.shape[1])
                break
        if best_v == K_eff:
            vals.extend([K_eff] * (r - step))
            break

        vals.append(best_v)
        bi = step + rel_i
        bj = step + rel_j
        if bi != step:
            A[[step, bi]] = A[[bi, step]]
        if bj != step:
            A[:, [step, bj]] = A[:, [bj, step]]

        pv = p_powers[best_v]
        mod_red = mod // pv
        piv_unit = int(A[step, step] // pv) % mod_red
        inv_unit = pow(piv_unit, -1, mod_red)

        if step + 1 < r:
            col_below = A[step + 1:, step]
            nz = np.nonzero(col_below)[0]
            if nz.size > 0:
                rows_idx = step + 1 + nz
                mults = ((col_below[nz] // pv) * inv_unit) % mod_red
                A[rows_idx, step + 1:] = (A[rows_idx, step + 1:] - np.outer(mults, A[step, step + 1:])) % mod
                A[rows_idx, step] = 0

    vals.sort()
    assert vals[-1] < K_eff, f"{label} (p={p}): PRECISION EXCEEDED! max_val={vals[-1]} >= K_eff={K_eff}"
    c = Counter(vals)
    s = " * ".join(f"{p}^{v}({cnt})" if v > 0 else f"1({cnt})" for v, cnt in sorted(c.items()))
    log(f"    [SNF {label}, p={p}] rk={r}, v_{p}(det)={sum(vals)}, max_v={vals[-1]}<{K_eff}: {s} ({time.time()-t0:.2f}s)")
    return vals


def run_for_prime(args):
    p, K, Psi_L, A2_I1, A2_I2, A2_LIN, A2_DEC20, A2_J11 = args
    log(f"\n=== Prime p = {p} (K = {K}, mod = {p**K}) ===")
    mod = int(p ** K)
    res = {}

    res["L"] = padic_snf_vals(Psi_L, p, K, label="L(2605)")
    assert Counter(res["L"]) == Counter({0: 2604, 1: 1}), res["L"][-5:]

    X_S, m_S = padic_left_kernel(A2_I1, p, K, expected_rank=854, label="S(1751)")
    K_S = K - m_S
    X_S_Psi = matmul_mod(X_S, Psi_L, mod)
    G_S = matmul_mod(X_S_Psi, X_S.T, mod)
    res["S"] = padic_snf_vals(G_S, p, K_S, label="S(1751)")

    X_T, m_T = padic_left_kernel(X_S_Psi.T, p, K_S, expected_rank=1751, label="T(854)")
    K_T = K_S - m_T
    G_T = matmul_mod(matmul_mod(X_T, Psi_L, mod), X_T.T, mod)
    res["T"] = padic_snf_vals(G_T, p, K_T, label="T(854)")

    U_ST = np.vstack([X_S, X_T]) % int(p ** K_T)
    res["G_ST"] = padic_snf_vals(U_ST, p, K_T, label="Glue L/(S+T)")

    M_S_I2 = matmul_mod(X_S, A2_I2, mod)
    U_V, m_V = padic_left_kernel(M_S_I2, p, K_S, expected_rank=750, label="V(1001)")
    K_V = K_S - m_V
    G_V = matmul_mod(matmul_mod(U_V, G_S, mod), U_V.T, mod)
    res["V"] = padic_snf_vals(G_V, p, K_V, label="V(1001)")

    # Extract W directly from S to avoid precision loss through V
    M_S_LIN = matmul_mod(X_S, A2_LIN, mod)
    U_W, m_W = padic_left_kernel(M_S_LIN, p, K_S, expected_rank=1001, label="W(750)")
    K_W = K_S - m_W
    G_W = matmul_mod(matmul_mod(U_W, G_S, mod), U_W.T, mod)
    res["W"] = padic_snf_vals(G_W, p, K_W, label="W(750)")

    K_VW = min(K_V, K_W)
    U_VW = np.vstack([U_V, U_W]) % int(p ** K_VW)
    res["G_VW"] = padic_snf_vals(U_VW, p, K_VW, label="Glue S/(V+W)")

    X_W = matmul_mod(U_W, X_S, mod)
    M_W_J11 = matmul_mod(X_W, A2_J11, mod)
    Z_DEC, m_DEC = padic_left_kernel(M_W_J11, p, K_W, expected_rank=30, label="W_DEC20(720)")
    K_DEC = K_W - m_DEC
    G_DEC = matmul_mod(matmul_mod(Z_DEC, G_W, mod), Z_DEC.T, mod)
    res["W_DEC20"] = padic_snf_vals(G_DEC, p, K_DEC, label="W_DEC20(720)")

    M_W_DEC = matmul_mod(X_W, A2_DEC20, mod)
    Z_J11, m_J11 = padic_left_kernel(M_W_DEC, p, K_W, expected_rank=720, label="W_J11(30)")
    K_J11 = K_W - m_J11
    G_J11 = matmul_mod(matmul_mod(Z_J11, G_W, mod), Z_J11.T, mod)
    res["W_J11"] = padic_snf_vals(G_J11, p, K_J11, label="W_J11(30)")

    K_DJ = min(K_DEC, K_J11)
    Z_DJ = np.vstack([Z_DEC, Z_J11]) % int(p ** K_DJ)
    res["G_DJ"] = padic_snf_vals(Z_DJ, p, K_DJ, label="Glue W/(DEC20+J11)")

    assert sum(res["S"]) + sum(res["T"]) == sum(res["L"]) + 2 * sum(res["G_ST"]), "Identity L/(S+T) failed!"
    assert sum(res["V"]) + sum(res["W"]) == sum(res["S"]) + 2 * sum(res["G_VW"]), "Identity S/(V+W) failed!"
    assert sum(res["W_DEC20"]) + sum(res["W_J11"]) == sum(res["W"]) + 2 * sum(res["G_DJ"]), "Identity W/(DEC+J11) failed!"
    log(f"  [VERIFIED p={p}] All 3 gluing determinant identities hold")
    return p, res


def combine_snf(vals2, vals3):
    assert len(vals2) == len(vals3)
    ed = [(2 ** a) * (3 ** b) for a, b in zip(vals2, vals3)]
    c = Counter(ed)
    snf_str = " * ".join(f"{k}^{v}" for k, v in sorted(c.items()))
    non_triv = " * ".join(f"Z/{k}Z^{v}" for k, v in sorted(c.items()) if k > 1) or "trivial (0)"
    return {
        "rank": len(ed),
        "v2_det": int(sum(vals2)),
        "v3_det": int(sum(vals3)),
        "ed_counter": {str(k): int(v) for k, v in sorted(c.items())},
        "snf_str": snf_str,
        "group_str": non_triv,
    }


def main():
    from multiprocessing import Pool
    log("=== Exact p-adic lattice and gluing computation for X_6^4 ===")
    pf_proof = prove_prime_support_2_3()
    I, I_L, Psi_L = build_pham_and_slice()
    A2_I1, A2_I2, A2_LIN, A2_DEC20, A2_J11 = classify_forms_and_build_A2(I, I_L)

    tasks = [
        (2, 26, Psi_L, A2_I1, A2_I2, A2_LIN, A2_DEC20, A2_J11),
        (3, 17, Psi_L, A2_I1, A2_I2, A2_LIN, A2_DEC20, A2_J11),
    ]
    with Pool(2) as pool:
        res_map = dict(pool.map(run_for_prime, tasks))
    r2, r3 = res_map[2], res_map[3]

    log("\n" + "=" * 96)
    log("Exact Z-Smith normal forms and gluing groups for X_6^4")
    log("=" * 96)

    keys = [
        ("L", "L = H_4(X_6^4, Z)_0 (primitive homology)"),
        ("S", "S = Hdg^2(X_6^4, Z)_0 (primitive Hodge lattice)"),
        ("T", "T = S^perp (transcendental lattice)"),
        ("G_ST", "Glue L / (S + T)"),
        ("V", "V = (V_4^6)_0 (linear cycle lattice)"),
        ("W", "W = V^perp cap S (non-linear Hodge lattice)"),
        ("G_VW", "Glue S / (V + W)"),
        ("W_DEC20", "W_DEC20 (DECOMP_2_0, rank 720)"),
        ("W_J11", "W_J11 (JOIN_1_1, rank 30)"),
        ("G_DJ", "Glue W / (W_DEC20 + W_J11)"),
    ]

    final = {"prime_support_proof": pf_proof, "lattices": {}}
    for k, title in keys:
        comb = combine_snf(r2[k], r3[k])
        final["lattices"][k] = comb
        if k.startswith("G_"):
            log(f"  {title:52s} | order = 2^{comb['v2_det']} * 3^{comb['v3_det']}")
            log(f"      group = {comb['group_str']}")
        else:
            log(f"  {title:52s} | rk={comb['rank']:4d}, |det| = 2^{comb['v2_det']} * 3^{comb['v3_det']}")
            log(f"      SNF   = {comb['snf_str']}")

    # Cross-check V against Aljovin-Movasati-Villaflor Loyola (2019), Table 1
    EXPECT_V_PRIM = {
        "1": 308, "3": 272, "6": 41, "12": 163,
        "36": 116, "108": 61, "216": 24, "648": 16,
    }
    assert final["lattices"]["V"]["ed_counter"] == EXPECT_V_PRIM, (
        final["lattices"]["V"]["ed_counter"], EXPECT_V_PRIM
    )
    log("\nVerified: SNF(V) matches Aljovin-Movasati-Villaflor Loyola Table 1.")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sextic_full_lattice_results.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(final, fp, indent=2)
    log(f"Saved: {out_path}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
