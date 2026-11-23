"""
Verification for Section 6.3(2-3):
  1. Checks S_6 permutation sign consistency against the canonical Jacobian sign.
  2. Computes the mod-3 rank of the 6 cosets of Z_DEC20^C4 together with planes P
     in the four-odd sector 00111 (rank 70), optionally cross-checking containment and
     span against K3.npy.xz from Santana de Jesus (2026).
"""
import itertools
import lzma
import os
import time
import numpy as np
from sage.all import CyclotomicField, PolynomialRing, QQ, matrix

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.2f}s] {msg}", flush=True)

K0 = CyclotomicField(12)
eta = K0.gen()
z6 = eta ** 2
L = K0.extension(PolynomialRing(K0, "t").gen() ** 3 - 2, "lam")
lam = L.gen()
I_ = L(eta ** 3)
sqrt3 = L(eta + eta ** 11)
a_c = -L(1) / lam

Rx = PolynomialRing(L, "x0,x1,x2,x3,x4,x5")
x = Rx.gens()
F6 = sum(v ** 6 for v in x)
MONS = [m for m in itertools.product(range(5), repeat=6) if sum(m) == 12]
NH = len(MONS)
mon_index = {m: i for i, m in enumerate(MONS)}
A6 = [tuple(v + 1 for v in m) for m in MONS]
pos = np.array([sum((a[k] - 1) * 5 ** (5 - k) for k in range(1, 6)) for a in A6])
neg = np.array([mon_index[tuple(4 - v for v in m)] for m in MONS])

def functional(fs, gs):
    assert sum(f * g for f, g in zip(fs, gs)) == F6
    det = matrix(Rx, [[f.derivative(v) for v in x] for f in list(fs) + list(gs)]).det()
    out = {}
    for mon, c in det.dict().items():
        mon = tuple(mon)
        if max(mon) <= 4:
            out[mon_index[tuple(4 - v for v in mon)]] = c
    return out

def res_pair(p1, p2):
    s = L(0)
    for i, c in p1.items():
        j = int(neg[i])
        if j in p2:
            s += c * p2[j]
    return -s / 6**7

def geom6(u, v, e):
    return sum(u**(5 - k) * ((e * v)**k) for k in range(6))

fP0 = [x[0] - eta*x[1], x[2] - eta*x[3], x[4] - eta*x[5]]
gP0 = [geom6(x[0], x[1], eta), geom6(x[2], x[3], eta), geom6(x[4], x[5], eta)]
Phi_P0 = functional(fP0, gP0)

def build_C4_direct(p0, p1, p2, p3, p4, p5):
    P1 = x[p0]**2 + I_*x[p0]*x[p1] + x[p1]**2
    P2 = x[p0]**2 - I_*x[p0]*x[p1] + x[p1]**2
    fs = [x[p2]**2 - a_c*P1, x[p3]**2 - a_c*P2, x[p4] - eta*x[p5]]
    gs = [x[p2]**4 + x[p2]**2*(a_c*P1) + (a_c*P1)**2,
          x[p3]**4 + x[p3]**2*(a_c*P2) + (a_c*P2)**2,
          geom6(x[p4], x[p5], eta)]
    return functional(fs, gs)

def perm_action(Phi, perm):
    inv = sum(1 for i in range(6) for j in range(i+1, 6) if perm[i] > perm[j])
    sgn = 1 if inv % 2 == 0 else -1
    out = {}
    for i, c in Phi.items():
        b = MONS[i]
        nb = [0]*6
        for k in range(6): nb[perm[k]] = b[k]
        out[mon_index[tuple(nb)]] = sgn * c
    return out, sgn

Phi_C4_id = build_C4_direct(0, 1, 2, 3, 4, 5)
log("=== Check 1: Permutation sign consistency for <Z_C4, P0> ===")
for perm in [(0,1,2,3,4,5), (0,3,1,2,4,5), (3,4,1,2,0,5), (3,5,1,2,0,4), (4,5,1,2,0,3)]:
    Phi_dir = build_C4_direct(*perm)
    Phi_act, sgn = perm_action(Phi_C4_id, perm)
    eq_plus = all(Phi_dir.get(k, 0) == Phi_act.get(k, 0) for k in set(Phi_dir) | set(Phi_act))
    eq_minus = all(Phi_dir.get(k, 0) == -Phi_act.get(k, 0) for k in set(Phi_dir) | set(Phi_act))
    ip_dir = res_pair(Phi_dir, Phi_P0)
    ip_act = res_pair(Phi_act, Phi_P0)
    log(f"  perm={perm} (sgn={sgn:+d}): dir==+act? {eq_plus}, dir==-act? {eq_minus} | "
        f"<dir,P0>+4/6 = {ip_dir + QQ(4)/6} | <act,P0>+4/6 = {ip_act + QQ(4)/6}")

log("\n=== Check 2: Sector 00111 (rank 70) mod-3 span ===")
import sys
k3_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get(
        "RAFATY_K3_PATH",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "external",
            "integral-hodge-fermat-sextic",
            "verifier",
            "data",
            "g13",
            "K3.npy.xz",
        ),
    )
)
K3 = None
if os.path.exists(k3_path):
    with lzma.open(k3_path, "rb") as f:
        K3 = np.load(f)
    log(f"Loaded Santana de Jesus K3 basis from {k3_path}: shape={K3.shape}")
else:
    log(
        f"[Note] Optional external K3.npy.xz not found at {k3_path}. "
        "Running self-contained rank check for P and Z_DEC20^C4; "
        "pass verifier/data/g13/K3.npy.xz as argv[1] or RAFATY_K3_PATH to also verify containment in K3_s."
    )

V1 = np.eye(5, dtype=np.int64)
for j in range(4): V1[j, j + 1] = -1
Vk = V1
for _ in range(4): Vk = np.kron(Vk, V1)
Psi = Vk + Vk.T; del Vk

def alpha(a):
    n = [sum(1 for v in a if v == j) for j in range(6)]
    q, r = divmod(-n[1] + 3 * n[4] + 4 * n[5], 3)
    e3 = n[1] - n[4] - n[5]
    s = {1: L(1) / 2, 2: sqrt3 / 2, 3: L(1), 4: sqrt3 / 2, 5: L(1) / 2}[a[0]]
    return L(2) ** q * lam ** r * L(3) ** (e3 // 2) * (sqrt3 if e3 % 2 else 1) * s * L(eta) ** (sum(a[1:]) % 12)

def prodz(a):
    out = L(1)
    for v in a[1:]: out *= L(z6) ** v - 1
    return out

KAPPA = QQ(3) / 2
rho = [1 / (KAPPA * alpha(a) * prodz(a)) for a in A6]
rho_e = [[j for j in range(3) if r_.list()[j] != 0][0] for r_ in rho]
rho_u = [r_.list()[e] for r_, e in zip(rho, rho_e)]
F1 = matrix(K0, 5, 5, [[z6 ** (a * b) for b in range(5)] for a in range(1, 6)])
G = F1.inverse()
def zpair(k):
    c = K0(k).list(); return c[0], c[2]
Gr = np.array([[int(zpair(6 * G[i, j])[0]) for j in range(5)] for i in range(5)], dtype=np.int64)
Gi = np.array([[int(zpair(6 * G[i, j])[1]) for j in range(5)] for i in range(5)], dtype=np.int64)

def to_v_int(Phi):
    re = np.zeros(NH, dtype=np.int64); im = np.zeros(NH, dtype=np.int64)
    for i, c in Phi.items():
        lst = c.list(); e = rho_e[i]; j0 = (-e) % 3
        cc = (rho_u[i] * lst[j0] * (2 if j0 + e == 3 else 1)).list()
        re[i], im[i] = int(2 * QQ(cc[0])), int(2 * QQ(cc[2]))
    T_r = np.zeros(3125, dtype=np.int64); T_i = np.zeros(3125, dtype=np.int64)
    T_r[pos] = re; T_i[pos] = im
    A_r = T_r.reshape((5,5,5,5,5)).astype(object); A_i = T_i.reshape((5,5,5,5,5)).astype(object)
    Gr_o = Gr.astype(object); Gi_o = Gi.astype(object)
    for ax in range(5):
        A_r = np.moveaxis(A_r, ax, 0).reshape(5, -1); A_i = np.moveaxis(A_i, ax, 0).reshape(5, -1)
        nr = Gr_o @ A_r - Gi_o @ A_i; ni = Gr_o @ A_i + Gi_o @ A_r + Gi_o @ A_i
        A_r = np.moveaxis(nr.reshape(5,5,5,5,5), 0, ax); A_i = np.moveaxis(ni.reshape(5,5,5,5,5), 0, ax)
    v_s = Psi.astype(object) @ A_r.reshape(3125)
    den = 2 * (6 ** 5)
    return np.array([int(val // den) for val in v_s], dtype=np.int64)

T1 = np.zeros((5, 5), dtype=np.int64)
for j in range(4): T1[j + 1, j] = 1
T1[:, 4] = -1
T1_3 = T1 @ T1 @ T1; I5 = np.eye(5, dtype=np.int64)
T1_pows = [I5]
for _ in range(5): T1_pows.append((T1_pows[-1] @ T1) % 3)

def project_sector(vecs_rows, s_bits):
    N = vecs_rows.shape[0]
    A = vecs_rows.reshape((N, 5, 5, 5, 5, 5))
    for ax in range(5):
        M_op = (I5 + ((-1) ** s_bits[ax]) * T1_3)
        A = np.tensordot(A, M_op, axes=([ax + 1], [0]))
        A = np.moveaxis(A, -1, ax + 1)
    return A.reshape((N, 3125))

def apply_twists_mod3(v0, shifts_list):
    out = []
    v_t = (v0 % 3).reshape((5,5,5,5,5))
    for sh in shifts_list:
        A = v_t
        for ax in range(5):
            if sh[ax] != 0:
                A = np.tensordot(A, T1_pows[sh[ax]], axes=([ax], [0]))
                A = np.moveaxis(A, -1, ax)
        out.append((A.reshape(3125)) % 3)
    return np.array(out, dtype=np.int64)

def rank_mod3(M):
    A = (M % 3).copy()
    nr, nc = A.shape; rk = 0
    for c in range(nc):
        col = A[rk:, c]; nz = np.nonzero(col)[0]
        if nz.size == 0: continue
        pi = rk + int(nz[0])
        if pi != rk: A[[rk, pi]] = A[[pi, rk]]
        A[rk, c:] = (A[rk, c:] * int(A[rk, c])) % 3
        below = A[rk+1:, c]; bnz = np.nonzero(below)[0]
        if bnz.size > 0:
            ri = rk + 1 + bnz
            A[ri, c:] = (A[ri, c:] - np.outer(below[bnz], A[rk, c:])) % 3
        rk += 1
    return rk

v_P0 = to_v_int(Phi_P0)
# 6 cosets of Z_DEC20^C4 in sector 00111 (even pair {1,2}, odd {0,3,4,5})
shifts_all = list(itertools.product([0, 1, 2], repeat=5))
c4_all6 = []
for p0_, p1_ in itertools.combinations([0, 3, 4, 5], 2):
    p4_, p5_ = [k for k in [0, 3, 4, 5] if k not in (p0_, p1_)]
    Phi_d = build_C4_direct(p0_, p1_, 1, 2, p4_, p5_)
    ip = res_pair(Phi_d, Phi_P0)
    if (ip + QQ(4)/6).denominator() != 1:
        assert (-ip + QQ(4)/6).denominator() == 1
        Phi_d = {k: -v for k, v in Phi_d.items()}
    v_d = to_v_int(Phi_d) - 4 * v_P0
    v_ds = project_sector(v_d.reshape(1, 3125), (0, 0, 1, 1, 1)) % 3
    c4_all6.append(apply_twists_mod3(v_ds[0], shifts_all))

C4_6 = np.vstack(c4_all6) % 3
log(f"Z_DEC20^C4 (6 cosets): sector 00111 mod-3 rank = {rank_mod3(C4_6)}/70")

plane_vecs = []
for p_x0 in [1, 2, 3, 4, 5]:
    rem = [k for k in [1, 2, 3, 4, 5] if k != p_x0]
    for idx_b in range(1, 4):
        b1 = rem[idx_b]
        r2 = [k for k in rem if k not in (rem[0], b1)]
        fP_ = [x[0] - eta*x[p_x0], x[rem[0]] - eta*x[b1], x[r2[0]] - eta*x[r2[1]]]
        gP_ = [geom6(x[0], x[p_x0], eta), geom6(x[rem[0]], x[b1], eta), geom6(x[r2[0]], x[r2[1]], eta)]
        Phi_p = functional(fP_, gP_)
        ip = res_pair(Phi_p, Phi_P0)
        if (ip + QQ(1)/6).denominator() != 1:
            Phi_p = {k: -v for k, v in Phi_p.items()}
        vp = to_v_int(Phi_p) - v_P0
        vp_s = project_sector(vp.reshape(1, 3125), (0, 0, 1, 1, 1)) % 3
        plane_vecs.append(apply_twists_mod3(vp_s[0], shifts_all[:80]))
P_all = np.vstack(plane_vecs) % 3
rk_c4 = rank_mod3(C4_6)
rk_pc4 = rank_mod3(np.vstack([P_all, C4_6]))
log(f"Planes P + Z_DEC20^C4 (6 cosets): sector 00111 mod-3 rank = {rk_pc4}/70")
assert rk_c4 == 66 and rk_pc4 == 70, f"Unexpected mod-3 ranks: C4={rk_c4}, P+C4={rk_pc4}"

if K3 is not None:
    # Entries in {0, 1, 2}, so dot products <= 4 * 3125 = 12500 << 2^53 (exact in float64).
    K3_paired = np.rint((K3 % 3).astype(np.float64) @ (Psi % 3).astype(np.float64)).astype(np.int64) % 3
    K3_s = project_sector(K3_paired, (0, 0, 1, 1, 1)) % 3
    rk_K3 = rank_mod3(K3_s)
    rk_comb = rank_mod3(np.vstack([K3_s, P_all, C4_6]))
    log(f"K3.npy cross-check (Section 6.3(2)): rank(K3_s) = {rk_K3}/70, rank(K3_s + P + C4) = {rk_comb}/70")
    assert rk_K3 == 70 and rk_comb == 70, f"Containment/span mismatch against K3.npy: rk_K3={rk_K3}, rk_comb={rk_comb}"

print("\nALL CHECKS PASSED")
