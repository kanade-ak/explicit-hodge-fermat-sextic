"""
Verification of Theorem C(1): computes exact p-adic residual indices (p = 2, 3) of
<P - P0, Z_DEC20^C4 - 4*P0, Z_J11^Aoki> inside S = Hdg^2(X_6^4, Z)_0 and writes
data/x64_ihc_c4_results.json.
"""
import itertools
import json
import os
import random
import time
from multiprocessing import Pool
import numpy as np

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:6.2f}s] {msg}", flush=True)

TARGET_V_S = {2: 1699, 3: 1282}


def load_templates():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cycle_templates_mod_q_v2.json")
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    B12 = [tuple(b) for b in data["B12"]]
    idx12 = {b: i for i, b in enumerate(B12)}
    conj_idx = np.array([idx12[tuple(4 - u for u in b)] for b in B12], dtype=np.int64)
    return idx12, conj_idx, data["primes"][0], data["primes"][1]


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
        "C4": make_sparse_template(pdata["detC4"]),
    }
    if P0_vec is None:
        P0_vec = build_cycle_vec(tmap["P"][0], tmap["P"][1], (0, 1, 2, 3, 4, 5), (0, 0, 0, 0, 0, 0), idx12, z6_pows, q)
    M = np.empty((len(specs), 1751), dtype=np.int64)
    for i, (fam, perm, sh) in enumerate(specs):
        v = build_cycle_vec(tmap[fam][0], tmap[fam][1], perm, sh, idx12, z6_pows, q)
        if fam == "P":
            v = (v - P0_vec) % q
        elif fam == "C4":
            v = (v - 4 * P0_vec) % q  # deg 4 -> integral primitive class [Z_C4] - 4[P0]
        M[i, :] = v
    return M, P0_vec


def exact_cross_gram_fast(Mr1, Mc1_conj, p1, Mr2_chk=None, Mc2_conj=None, p2=None):
    q1 = p1["q"]
    half1 = q1 // 2
    G = (Mr1 @ Mc1_conj.T) % q1
    G = (G * p1["C_norm_mod_q"]) % q1
    G = np.where(G > half1, G - q1, G)
    if Mr2_chk is not None:
        q2 = p2["q"]
        half2 = q2 // 2
        G2 = (Mr2_chk @ Mc2_conj.T) % q2
        G2 = (G2 * p2["C_norm_mod_q"]) % q2
        G2 = np.where(G2 > half2, G2 - q2, G2)
        assert np.array_equal(G[:Mr2_chk.shape[0]], G2), "q1 vs q2 verification failed! (integrality error?)"
    return G


def padic_row_hnf_reduce(A_in, p, K, label=""):
    t0 = time.time()
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
    R_mat = A[:R, :].copy()
    max_v = max(diag_vals)
    assert max_v < K, f"{label} (p={p}): max_v={max_v} >= K={K}"
    log(f"    [p-HNF {label}, p={p}] {N}x{R} -> v_{p}(det)={sum(diag_vals)}, max_v={max_v}<{K} ({time.time()-t0:.2f}s)")
    return R_mat, diag_vals


def perm_shift(perm, sh):
    v = [0] * 6
    for k in range(6):
        v[perm[k]] = sh[k]
    return tuple(v)


def worker_prime(args):
    p, K, G_B, batches = args
    log(f"\n=== Theorem C(1): p = {p} (K = {K}) ===")
    R_cur, vals_GB = padic_row_hnf_reduce(G_B, p, K, label="0_G_B")
    vp_GB = sum(vals_GB)
    vp_S = TARGET_V_S[p]
    trace = [{"stage": "0_G_B", "vp_index": int((vp_GB - vp_S) // 2)}]
    for name, H_mat in batches:
        R_cur, vals_H = padic_row_hnf_reduce(np.vstack([R_cur, H_mat]), p, K, label=name)
        rem_idx = int(sum(vals_H) - (vp_GB + vp_S) // 2)
        assert rem_idx >= 0
        log(f"    [p={p}] after {name}: v_{p}([S : Lambda]) = {rem_idx}")
        trace.append({"stage": name, "vp_index": rem_idx})
    return str(p), {"p": p, "vp_S": vp_S, "vp_GB": int(vp_GB), "final_vp_index": trace[-1]["vp_index"], "trace": trace}


def main():
    log("=== Theorem C(1): Z[1/2]-IHC verification for <P, Z_DEC20^C4, Z_J11^Aoki> ===")
    idx12, conj_idx, p1, p2 = load_templates()

    p_perms = []
    for b1 in range(1, 6):
        r1 = [x for x in range(1, 6) if x != b1]
        b2 = r1[0]
        for b3 in r1[1:]:
            r2 = [x for x in r1[1:] if x != b3]
            p_perms.append((0, b1, b2, b3, r2[0], r2[1]))
    # 90 cosets stabilizing {{0,1}, {2,3}, {4,5}}
    c4_perms = []
    for p0, p1_ in itertools.combinations(range(6), 2):
        rem = [x for x in range(6) if x not in (p0, p1_)]
        for p2_, p3_ in itertools.combinations(rem, 2):
            p4_, p5_ = [x for x in rem if x not in (p2_, p3_)]
            c4_perms.append((p0, p1_, p2_, p3_, p4_, p5_))
    assert len(c4_perms) == 90
    j_perms = c4_perms

    planes_nz = []
    for perm in p_perms:
        for a0, a1, a2 in itertools.product(range(6), repeat=3):
            sh = [0] * 6
            sh[perm[1]] = a0; sh[perm[3]] = a1; sh[perm[5]] = a2
            planes_nz.append(("P", perm, tuple(sh)))
    planes_nz = planes_nz[1:]

    rng = random.Random(2026112303)
    def gen_C4(n):
        return [("C4", perm, perm_shift(perm, (rng.randrange(6), rng.randrange(6), rng.randrange(3), rng.randrange(3), rng.randrange(6), 0)))
                for perm in c4_perms for _ in range(n)]
    def gen_J(n):
        return [("J", perm, perm_shift(perm, (rng.randrange(6), rng.randrange(6), rng.randrange(3), rng.randrange(3), rng.randrange(3), 0)))
                for perm in j_perms for _ in range(n)]

    C4_b1 = gen_C4(80)
    C4_b2 = gen_C4(80)
    J_b1 = gen_J(60)

    MP1, P0_1 = materialize(planes_nz, p1, idx12)
    _, P0_2 = materialize(planes_nz[:1], p2, idx12)
    MC1, _ = materialize(C4_b1, p1, idx12, P0_1)
    MJ1, _ = materialize(J_b1, p1, idx12, P0_1)

    q1 = p1["q"]
    pool_specs = planes_nz + C4_b1 + J_b1
    M_pool = np.vstack([MP1, MC1, MJ1])
    piv_rows = []; piv_cols = []
    ech = np.zeros((1751, 1751), dtype=np.int64)
    for i in range(M_pool.shape[0]):
        v = M_pool[i].copy()
        for r_i, c_i in enumerate(piv_cols):
            val = v[c_i]
            if val != 0:
                v = (v - val * ech[r_i]) % q1
        nz = np.nonzero(v)[0]
        if nz.size == 0:
            continue
        c = int(nz[0])
        ech[len(piv_rows)] = (v * pow(int(v[c]), -1, q1)) % q1
        piv_rows.append(i); piv_cols.append(c)
        if len(piv_rows) == 1751:
            break
    assert len(piv_rows) == 1751, f"rank = {len(piv_rows)} < 1751!"
    log("Extracted rank-1751 basis (P + Z_DEC20^C4 + Z_J11^Aoki)")

    B_specs = [pool_specs[i] for i in piv_rows]
    MB1 = M_pool[piv_rows]
    MB2, _ = materialize(B_specs, p2, idx12, P0_2)
    MB1_conj = MB1[:, conj_idx]
    MB2_conj = MB2[:, conj_idx]
    G_B = exact_cross_gram_fast(MB1, MB1_conj, p1, MB2, MB2_conj, p2)
    log("Exact integer G_B restored and verified across q1, q2")

    def build_H(name, specs, M1_pre=None):
        M1 = M1_pre if M1_pre is not None else materialize(specs, p1, idx12, P0_1)[0]
        M2_chk, _ = materialize(specs[:30], p2, idx12, P0_2)
        H = exact_cross_gram_fast(M1, MB1_conj, p1, M2_chk, MB2_conj, p2)
        log(f"  Batch {name}: {H.shape}")
        return (name, H)

    batches = [
        build_H("1_all_3239_planes", planes_nz, MP1),
        build_H("2_Z_C4_7200_r1", C4_b1, MC1),
        build_H("3_Z_C4_7200_r2", C4_b2),
        build_H("4_Z_J11_5400_r1", J_b1, MJ1),
    ]

    with Pool(2) as pool:
        res_map = dict(pool.map(worker_prime, [
            (2, 26, G_B, batches),
            (3, 17, G_B, batches),
        ]))

    log("\n" + "=" * 88)
    log("Summary (<P, Z_DEC20^C4, Z_J11^Aoki> <= S):")
    log("=" * 88)
    for p_s in ("2", "3"):
        r = res_map[p_s]
        log(f"  p = {p_s}: final v_{p_s}([S : Lambda_0]) = {r['final_vp_index']}")
        for step in r["trace"]:
            log(f"      {step['stage']:22s} -> v_{p_s} = {step['vp_index']}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "x64_ihc_c4_results.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(res_map, fp, indent=2)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
