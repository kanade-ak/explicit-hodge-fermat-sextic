"""
Layer-by-layer orthogonal projection diagnostic decomposing the residual index
2^426 * 3^90 of <P, Z_DEC20^Aoki, Z_J11^Aoki> across V, W_DEC20, and W_J11.
"""
import itertools
import json
import os
import random
import time
from collections import Counter
import numpy as np

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.2f}s] {msg}", flush=True)


def classify_beta(b):
    a = tuple(u + 1 for u in b)
    c = Counter(a)
    if all((cnt % 2 == 0 if x == 3 else c.get(6 - x, 0) == cnt) for x, cnt in c.items()):
        return "LINEAR"
    if any((6 - x) in c and (x != 3 or c[x] >= 2) for x in c):
        return "DECOMP_2_0"
    return "JOIN_1_1"


def load_templates():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cycle_templates_mod_q.json")
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    B12 = [tuple(b) for b in data["B12"]]
    idx12 = {b: i for i, b in enumerate(B12)}
    conj_idx = np.array([idx12[tuple(4 - u for u in b)] for b in B12], dtype=np.int64)
    cats = [classify_beta(b) for b in B12]
    cols_L = {
        "LINEAR": np.array([i for i, c in enumerate(cats) if c == "LINEAR"], dtype=np.int64),
        "DECOMP_2_0": np.array([i for i, c in enumerate(cats) if c == "DECOMP_2_0"], dtype=np.int64),
        "JOIN_1_1": np.array([i for i, c in enumerate(cats) if c == "JOIN_1_1"], dtype=np.int64),
    }
    return idx12, conj_idx, cols_L, data["primes"][0], data["primes"][1]


def make_sparse_template(enc_list):
    return (np.array([x[0] for x in enc_list], dtype=np.int64),
            np.array([x[1] for x in enc_list], dtype=np.int64))


def build_cycle_vec(exps, coeffs, perm, shifts, idx12, z6_pows, q):
    inv = sum(1 for i in range(6) for j in range(i + 1, 6) if perm[i] > perm[j])
    sgn = 1 if (inv % 2 == 0) else (q - 1)
    v = np.zeros(1751, dtype=np.int64)
    for k in range(exps.shape[0]):
        b = exps[k]
        nb = [0] * 6
        nb[perm[0]] = int(b[0]); nb[perm[1]] = int(b[1]); nb[perm[2]] = int(b[2])
        nb[perm[3]] = int(b[3]); nb[perm[4]] = int(b[4]); nb[perm[5]] = int(b[5])
        nb_t = tuple(nb)
        pw = (shifts[0] * (nb[0] + 1) + shifts[1] * (nb[1] + 1) + shifts[2] * (nb[2] + 1) +
              shifts[3] * (nb[3] + 1) + shifts[4] * (nb[4] + 1) + shifts[5] * (nb[5] + 1)) % 6
        v[idx12[nb_t]] = (v[idx12[nb_t]] + coeffs[k] * sgn * z6_pows[pw]) % q
    return v


def materialize(specs, pdata, idx12, P0_vec=None):
    q = pdata["q"]
    z6_pows = [pow(pdata["z6"], k, q) for k in range(6)]
    tmap = {
        "P": make_sparse_template(pdata["detP"]),
        "D": make_sparse_template(pdata["detD_Aoki"]),
        "J": make_sparse_template(pdata["detJ_Aoki"]),
    }
    if P0_vec is None:
        P0_vec = build_cycle_vec(tmap["P"][0], tmap["P"][1], (0, 1, 2, 3, 4, 5), (0, 0, 0, 0, 0, 0), idx12, z6_pows, q)
    M = np.empty((len(specs), 1751), dtype=np.int64)
    for i, (fam, perm, sh) in enumerate(specs):
        v = build_cycle_vec(tmap[fam][0], tmap[fam][1], perm, sh, idx12, z6_pows, q)
        if fam == "P":
            v = (v - P0_vec) % q
        M[i, :] = v
    return M, P0_vec


def perm_shift(perm, sh):
    v = [0] * 6
    for k in range(6):
        v[perm[k]] = sh[k]
    return tuple(v)


def exact_cross_gram_layer(Mr1, Mc1_conj, p1, cols, scale=1, Mr2_chk=None, Mc2_conj=None, p2=None):
    q1 = p1["q"]
    half1 = q1 // 2
    G = (Mr1[:, cols] @ Mc1_conj[:, cols].T) % q1
    G = (G * ((p1["C_norm_mod_q"] * scale) % q1)) % q1
    G = np.where(G > half1, G - q1, G)
    if Mr2_chk is not None:
        q2 = p2["q"]
        half2 = q2 // 2
        G2 = (Mr2_chk[:, cols] @ Mc2_conj[:, cols].T) % q2
        G2 = (G2 * ((p2["C_norm_mod_q"] * scale) % q2)) % q2
        G2 = np.where(G2 > half2, G2 - q2, G2)
        assert np.array_equal(G[:Mr2_chk.shape[0]], G2), "q1 vs q2 mismatch!"
    return G


def padic_hnf(A_in, p, K):
    N, R = A_in.shape
    mod = int(p ** K)
    p_powers = [int(p ** j) for j in range(K + 1)]
    A = (A_in % mod).copy()
    diag_vals = []
    for c in range(R):
        col_sub = A[c:, c]
        best_v = K
        rel_i = -1
        for v in range(K):
            mask = (col_sub % p_powers[v + 1]) != 0
            if np.any(mask):
                best_v = v
                rel_i = int(np.argmax(mask))
                break
        if best_v == K:
            diag_vals.append(K)
            continue
        diag_vals.append(best_v)
        bi = c + rel_i
        if bi != c:
            A[[c, bi]] = A[[bi, c]]
        pv = p_powers[best_v]
        mod_red = mod // pv
        inv_unit = pow(int(A[c, c] // pv) % mod_red, -1, mod_red)
        A[c, c:] = (A[c, c:] * inv_unit) % mod
        if c + 1 < N:
            below = A[c + 1:, c]
            nz = np.nonzero(below)[0]
            if nz.size > 0:
                ri = c + 1 + nz
                mults = (below[nz] // pv) % mod_red
                A[ri, c:] = (A[ri, c:] - np.outer(mults, A[c, c:])) % mod
    assert max(diag_vals) < K
    return A[:R, :].copy(), diag_vals


def padic_snf_vals(A_in, p, K):
    """Return diagonal p-valuation counts of SNF(A_in) mod p^K."""
    R = A_in.shape[0]
    mod = int(p ** K)
    p_powers = [int(p ** j) for j in range(K + 1)]
    A = (A_in % mod).copy()
    vals = []
    for k in range(R):
        sub = A[k:, k:]
        best_v = K
        bi = bj = -1
        for v in range(K):
            mask = (sub % p_powers[v + 1]) != 0
            if np.any(mask):
                best_v = v
                idx = int(np.argmax(mask))
                bi, bj = divmod(idx, R - k)
                break
        if best_v == K:
            vals.append(K)
            continue
        vals.append(best_v)
        if bi != 0:
            A[[k, k + bi]] = A[[k + bi, k]]
        if bj != 0:
            A[:, [k, k + bj]] = A[:, [k + bj, k]]
        pv = p_powers[best_v]
        mod_red = mod // pv
        inv_u = pow(int(A[k, k] // pv) % mod_red, -1, mod_red)
        A[k, k:] = (A[k, k:] * inv_u) % mod
        if k + 1 < R:
            below = A[k + 1:, k]
            nz = np.nonzero(below)[0]
            if nz.size > 0:
                ri = k + 1 + nz
                mults = (below[nz] // pv) % mod_red
                A[ri, k:] = (A[ri, k:] - np.outer(mults, A[k, k:])) % mod
            A[k, k + 1:] = 0
    assert max(vals) < K
    return Counter(vals)


def analyze_layer(name, M1_gens, M2_chk, conj_idx, cols, p1, p2, scale, target_vp_W, target_vp_projS):
    """Compute Gram matrix and p-adic index of pi_L(Lambda_alg); scale=3 clears 3-adic glue denominators."""
    q1 = p1["q"]
    r = len(cols)
    M_sub = M1_gens[:, cols]
    piv_rows = []
    piv_cols = []
    ech = np.zeros((r, r), dtype=np.int64)
    for i in range(M_sub.shape[0]):
        v = M_sub[i].copy()
        for r_i, c_i in enumerate(piv_cols):
            val = v[c_i]
            if val != 0:
                v = (v - val * ech[r_i]) % q1
        nz = np.nonzero(v)[0]
        if nz.size == 0:
            continue
        c = int(nz[0])
        ech[len(piv_rows)] = (v * pow(int(v[c]), -1, q1)) % q1
        piv_rows.append(i)
        piv_cols.append(c)
        if len(piv_rows) == r:
            break
    assert len(piv_rows) == r, f"{name}: rank {len(piv_rows)} != {r}"

    MB1 = M1_gens[piv_rows]
    MB1_conj = MB1[:, conj_idx]
    G_B = exact_cross_gram_layer(MB1, MB1_conj, p1, cols, scale=scale)
    H = exact_cross_gram_layer(M1_gens, MB1_conj, p1, cols, scale=scale,
                               Mr2_chk=M2_chk, Mc2_conj=None, p2=None)

    log(f"\n--- Layer {name} (rk = {r}, scale = {scale}) ---")
    for p, K in [(2, 26), (3, 17)]:
        _, vGB = padic_hnf(G_B, p, K)
        R_H, vH = padic_hnf(H, p, K)
        # det(pi_L(Lambda)) = det(R_H)^2 / (det(G_B) * scale^r)
        vp_scale = (1 if (scale == 3 and p == 3) else 0)
        vp_det_proj_alg = 2 * sum(vH) - sum(vGB) - r * vp_scale
        idx_in_projS = (vp_det_proj_alg - target_vp_projS[p]) // 2
        idx_vs_W = (vp_det_proj_alg - target_vp_W[p]) // 2
        log(f"  p={p}: v_{p}(det pi_L(Lambda_alg)) = {vp_det_proj_alg} "
            f"(W_L={target_vp_W[p]}, pi_L(S)={target_vp_projS[p]}) "
            f"==> v_{p}([pi_L(S) : pi_L(Lambda_alg)]) = {idx_in_projS}  (vs W_L: {idx_vs_W})")


def main():
    log("=== Layer-by-layer cokernel decomposition for <P, Z_DEC20^Aoki, Z_J11^Aoki> ===")
    idx12, conj_idx, cols_L, p1, p2 = load_templates()

    p_perms = []
    for b1 in range(1, 6):
        r1 = [x for x in range(1, 6) if x != b1]
        b2 = r1[0]
        for b3 in r1[1:]:
            r2 = [x for x in r1[1:] if x != b3]
            p_perms.append((0, b1, b2, b3, r2[0], r2[1]))
    d_perms = []
    for p0 in range(6):
        for p1_ in range(6):
            if p1_ == p0:
                continue
            rem = [x for x in range(6) if x not in (p0, p1_)]
            for p2_, p3_ in itertools.combinations(rem, 2):
                p4_, p5_ = [x for x in rem if x not in (p2_, p3_)]
                d_perms.append((p0, p1_, p2_, p3_, p4_, p5_))
    j_perms = []
    for p0 in range(6):
        for p1_ in range(p0 + 1, 6):
            rem = [x for x in range(6) if x not in (p0, p1_)]
            for p2_, p3_ in itertools.combinations(rem, 2):
                p4_, p5_ = [x for x in rem if x not in (p2_, p3_)]
                j_perms.append((p0, p1_, p2_, p3_, p4_, p5_))

    planes_nz = []
    for perm in p_perms:
        for a0, a1, a2 in itertools.product(range(6), repeat=3):
            sh = [0] * 6
            sh[perm[1]] = a0; sh[perm[3]] = a1; sh[perm[5]] = a2
            planes_nz.append(("P", perm, tuple(sh)))
    planes_nz = planes_nz[1:]

    rng = random.Random(2026112302)
    D_b1 = [("D", perm, perm_shift(perm, (rng.randrange(6), rng.randrange(2), rng.randrange(3), rng.randrange(3), rng.randrange(6), 0)))
            for perm in d_perms for _ in range(40)]
    J_b1 = [("J", perm, perm_shift(perm, (rng.randrange(6), rng.randrange(6), rng.randrange(3), rng.randrange(3), rng.randrange(3), 0)))
            for perm in j_perms for _ in range(60)]

    MP1, P0_1 = materialize(planes_nz, p1, idx12)
    MD1, _ = materialize(D_b1, p1, idx12, P0_1)
    MJ1, _ = materialize(J_b1, p1, idx12, P0_1)
    M_all = np.vstack([MP1, MD1, MJ1])
    M_DJ = np.vstack([MD1, MJ1])
    log(f"Materialized cycles: M_all = {M_all.shape}")

    # Target valuations from sextic_full_lattice_results.json (S/(V+W) = (Z/3)^375, zero 2-adic glue)
    analyze_layer("V (planes only)", MP1, None, conj_idx, cols_L["LINEAR"], p1, p2, 1,
                  {2: 841, 3: 1027}, {2: 841, 3: 1027})
    analyze_layer("V (all cycles, pi_V)", M_all, None, conj_idx, cols_L["LINEAR"], p1, p2, 3,
                  {2: 841, 3: 1027}, {2: 841, 3: 1027 - 750})

    cols_W = np.concatenate([cols_L["DECOMP_2_0"], cols_L["JOIN_1_1"]])
    analyze_layer("W (D+J, pi_W)", M_DJ, None, conj_idx, cols_W, p1, p2, 3,
                  {2: 858, 3: 1005}, {2: 858, 3: 1005 - 750})

    analyze_layer("DEC20 (D only)", MD1, None, conj_idx, cols_L["DECOMP_2_0"], p1, p2, 3,
                  {2: 800, 3: 990}, {2: 800, 3: 990 - 750})
    analyze_layer("DEC20 (D+J)", M_DJ, None, conj_idx, cols_L["DECOMP_2_0"], p1, p2, 3,
                  {2: 800, 3: 990}, {2: 800, 3: 990 - 750})

    analyze_layer("J11 (J only)", MJ1, None, conj_idx, cols_L["JOIN_1_1"], p1, p2, 3,
                  {2: 58, 3: 75}, {2: 58, 3: 75 - 60})

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
