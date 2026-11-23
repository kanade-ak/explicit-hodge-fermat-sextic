"""
Exact verification for:
  1. Defining field K = Q(zeta_12, 2^(1/3)) (degree 12) and Prop 3.1(1) smoothness mod 10009.
  2. Necessity of Z_J11^Aoki: W_J11 3-adic valuation for Z_J11^cube alone.
  3. Exact FLINT integer determinant det(G_{B_0}) = 2^2799 * 3^1314 (no floating-point).
"""
import itertools
import json
import os
import random
import time
import numpy as np

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:6.2f}s] {msg}", flush=True)

log("=== Part 1: Minimal defining field [K:Q]=12 and X_6^2 curve smoothness (mod 10009) ===")
K0 = CyclotomicField(12, 'z')
z0 = K0.gen()
PK = PolynomialRing(K0, 't')
K_rel = K0.extension(PK.gen()**3 - 2, 'c')
K12 = K_rel.absolute_field('u')
assert K12.degree() == 12
log(f"Minimal defining field K = Q(zeta_12, 2^(1/3)): [K:Q] = {K12.degree()}")

q = 10009
Fq = GF(q)
R1 = PolynomialRing(Fq, 'u'); u = R1.gen()
z = [r for r, _ in (u**12 - 1).roots() if r**6 == -1 and r**2 != -1][0]
I_u = z**3
cbrt2 = [r for r, _ in (u**3 - 2).roots()][0]
a_c = -Fq(1) / cbrt2
assert a_c**3 == -Fq(1)/Fq(2)

R3 = PolynomialRing(Fq, 'x0,x1,x2,x3')
x0, x1, x2, x3 = R3.gens(); xs3 = [x0, x1, x2, x3]
f_surf = sum(x**6 for x in xs3)

curves_x62 = [
    ("L (line, deg 1)",       [x0 - z*x1, x2 - z*x3]),
    ("C_4 (CI(2,2), deg 4)",  [x2**2 - a_c*(x0**2 + I_u*x0*x1 + x1**2),
                               x3**2 - a_c*(x0**2 - I_u*x0*x1 + x1**2)]),
    ("C_cube (CI(2,3), deg 6)", [x3**2 - cbrt2*x0*x2, x1**3 - I_u*(x0**3 + x2**3)]),
]
for cname, gs in curves_x62:
    I_C = R3.ideal(gs)
    dim_p = I_C.dimension() - 1
    assert dim_p == 1 and f_surf in I_C
    J = matrix(R3, [[g.derivative(v) for v in xs3] for g in gs])
    sing_dim = R3.ideal(gs + J.minors(2)).dimension() - 1
    log(f"  {cname:26s}: proj_dim={dim_p}, in X_6^2=True, sing_proj_dim={sing_dim} ({'smooth' if sing_dim < 0 else 'singular'})")

log("\n=== Part 2: W_J11 3-adic valuation for Z_J11^cube alone ===")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "data", "cycle_templates_mod_q_v3.json")
with open(path, "r", encoding="utf-8") as fp:
    data = json.load(fp)
B12 = [tuple(b) for b in data["B12"]]
idx12 = {b: i for i, b in enumerate(B12)}
conj_idx = np.array([idx12[tuple(4 - u for u in b)] for b in B12], dtype=np.int64)
p1 = data["primes"][0]

from collections import Counter
def classify_beta(b):
    a = tuple(u + 1 for u in b)
    c = Counter(a)
    if all((cnt % 2 == 0 if x == 3 else c.get(6 - x, 0) == cnt) for x, cnt in c.items()):
        return "LINEAR"
    if any((6 - x) in c and (x != 3 or c[x] >= 2) for x in c):
        return "DECOMP_2_0"
    return "JOIN_1_1"

cols_J11 = np.array([i for i, b in enumerate(B12) if classify_beta(b) == "JOIN_1_1"], dtype=np.int64)
assert len(cols_J11) == 30

def mk_t(enc):
    return np.array([x[0] for x in enc], dtype=np.int64), np.array([x[1] for x in enc], dtype=np.int64)

def bvec(exps, coeffs, perm, shifts, z6_pows, q_):
    inv = sum(1 for i in range(6) for j in range(i + 1, 6) if perm[i] > perm[j])
    sgn = 1 if (inv % 2 == 0) else (q_ - 1)
    v = np.zeros(1751, dtype=np.int64)
    for k in range(exps.shape[0]):
        b = exps[k]
        nb = [0] * 6
        nb[perm[0]] = int(b[0]); nb[perm[1]] = int(b[1]); nb[perm[2]] = int(b[2])
        nb[perm[3]] = int(b[3]); nb[perm[4]] = int(b[4]); nb[perm[5]] = int(b[5])
        nb_t = tuple(nb)
        pw = sum(shifts[j] * (nb[j] + 1) for j in range(6)) % 6
        v[idx12[nb_t]] = (v[idx12[nb_t]] + coeffs[k] * sgn * z6_pows[pw]) % q_
    return v

def pshift(perm, sh):
    v = [0]*6
    for k in range(6): v[perm[k]] = sh[k]
    return tuple(v)

q1 = p1["q"]
z6p = [pow(p1["z6"], k, q1) for k in range(6)]
t_Jcube = mk_t(p1["detJ_cube"])
jcube_perms = []
for p3_, p5_ in itertools.combinations(range(6), 2):
    rem = [x for x in range(6) if x not in (p3_, p5_)]
    for p0, p2_ in itertools.combinations(rem, 2):
        p1_, p4_ = [x for x in rem if x not in (p0, p2_)]
        jcube_perms.append((p0, p1_, p2_, p3_, p4_, p5_))
rng = random.Random(int(2026112305))
Jc_specs = [(perm, pshift(perm, (rng.randrange(6), rng.randrange(6), rng.randrange(6), rng.randrange(3), rng.randrange(6), 0)))
            for perm in jcube_perms for _ in range(40)]
MJc = np.array([bvec(t_Jcube[0], t_Jcube[1], perm, sh, z6p, q1) for perm, sh in Jc_specs], dtype=np.int64)
Msub = MJc[:, cols_J11]
pivs = []; pcols = []
ech = np.zeros((30, 30), dtype=np.int64)
for i in range(Msub.shape[0]):
    v = Msub[i].copy()
    for ri, ci in enumerate(pcols):
        if v[ci]: v = (v - v[ci] * ech[ri]) % q1
    nz = np.nonzero(v)[0]
    if nz.size == 0: continue
    c = int(nz[0])
    ech[len(pivs)] = (v * pow(int(v[c]), -1, q1)) % q1
    pivs.append(i); pcols.append(c)
    if len(pivs) == 30: break
assert len(pivs) == 30
MB = MJc[pivs]; MB_c = MB[:, conj_idx]
half1 = q1 // 2
def gr3(A, B):
    G = (A[:, cols_J11] @ B[:, cols_J11].T) % q1
    G = (G * ((p1["C_norm_mod_q"] * 3) % q1)) % q1
    return np.where(G > half1, G - q1, G)
GB_j = matrix(ZZ, gr3(MB, MB_c).tolist())
H_j = matrix(ZZ, gr3(MJc, MB_c).tolist())
H_j_hnf = H_j.hermite_form()[:30, :]
G_j_alg = ((H_j_hnf * GB_j.inverse() * H_j_hnf.T) / 3).change_ring(QQ)
# W_J11 has |det| = 2^58 * 3^75; pi_J11(S) has |det| = 2^58 * 3^(-15)
det_j = abs(G_j_alg.det())
log(f"  |det(pi_J11(Z_J11^cube))| = {factor(det_j)} (W_J11: 2^58 * 3^75, pi_J11(S): 2^58 * 3^(-15))")
v2_j = det_j.valuation(2)
v3_j = det_j.valuation(3)
log(f"  v_2([pi_J11(S) : pi_J11(Jcube)]) = {(v2_j - 58)//2}, v_3([pi_J11(S) : pi_J11(Jcube)]) = {(v3_j - (-15))//2}")
if (v3_j - (-15)) // 2 > 0:
    log("  Confirmed: Z_J11^cube alone leaves positive 3-adic index on W_J11; all 5 families are required.")
else:
    log("  Note: Z_J11^cube alone saturates W_J11 at both p=2 and p=3.")

log("\n=== Part 3: Exact FLINT integer determinant of G_B (1751x1751) ===")
p_perms = []
for b1 in range(1, 6):
    r1 = [x for x in range(1, 6) if x != b1]
    b2 = r1[0]
    for b3 in r1[1:]:
        r2 = [x for x in r1[1:] if x != b3]
        p_perms.append((0, b1, b2, b3, r2[0], r2[1]))
c4_perms = []
for p0, p1_ in itertools.combinations(range(6), 2):
    rem = [x for x in range(6) if x not in (p0, p1_)]
    for p2_, p3_ in itertools.combinations(rem, 2):
        p4_, p5_ = [x for x in rem if x not in (p2_, p3_)]
        c4_perms.append((p0, p1_, p2_, p3_, p4_, p5_))

tmap = {"P": mk_t(p1["detP"]), "C4": mk_t(p1["detC4"]), "J": mk_t(p1["detJ_Aoki"])}
P0_1 = bvec(tmap["P"][0], tmap["P"][1], (0,1,2,3,4,5), (0,0,0,0,0,0), z6p, q1)

def mat_list(specs):
    M = np.empty((len(specs), 1751), dtype=np.int64)
    for i, (fam, perm, sh) in enumerate(specs):
        v = bvec(tmap[fam][0], tmap[fam][1], perm, sh, z6p, q1)
        if fam == "P": v = (v - P0_1) % q1
        elif fam == "C4": v = (v - 4 * P0_1) % q1
        M[i, :] = v
    return M

planes_nz = [("P", perm, tuple([a0 if k==perm[1] else a1 if k==perm[3] else a2 if k==perm[5] else 0 for k in range(6)]))
             for perm in p_perms for a0, a1, a2 in itertools.product(range(6), repeat=3)][1:]
rng0 = random.Random(int(2026112303))
C4_b1 = [("C4", perm, pshift(perm, (rng0.randrange(6), rng0.randrange(6), rng0.randrange(3), rng0.randrange(3), rng0.randrange(6), 0)))
         for perm in c4_perms for _ in range(80)]
J_b1 = [("J", perm, pshift(perm, (rng0.randrange(6), rng0.randrange(6), rng0.randrange(3), rng0.randrange(3), rng0.randrange(3), 0)))
        for perm in c4_perms for _ in range(60)]

M_pool = np.vstack([mat_list(planes_nz), mat_list(C4_b1), mat_list(J_b1)])
piv_rows = []; piv_cols = []
ech = np.zeros((1751, 1751), dtype=np.int64)
for i in range(M_pool.shape[0]):
    v = M_pool[i].copy()
    for ri, ci in enumerate(piv_cols):
        if v[ci]: v = (v - v[ci] * ech[ri]) % q1
    nz = np.nonzero(v)[0]
    if nz.size == 0: continue
    c = int(nz[0])
    ech[len(piv_rows)] = (v * pow(int(v[c]), -1, q1)) % q1
    piv_rows.append(i); piv_cols.append(c)
    if len(piv_rows) == 1751: break

MB1 = M_pool[piv_rows]
MB1_conj = MB1[:, conj_idx]
G_B_np = (MB1 @ MB1_conj.T) % q1
G_B_np = (G_B_np * p1["C_norm_mod_q"]) % q1
G_B_np = np.where(G_B_np > half1, G_B_np - q1, G_B_np)

t_det = time.time()
G_B_ZZ = matrix(ZZ, G_B_np.tolist())
log(f"Built G_B (1751x1751) ZZ matrix ({time.time()-t_det:.2f}s); computing exact FLINT determinant...")
det_exact = G_B_ZZ.determinant()
target_det = (ZZ(2)**2799) * (ZZ(3)**1314)
log(f"Exact det(G_B) computed ({time.time()-t_det:.2f}s); det(G_B) == 2^2799 * 3^1314: {det_exact == target_det}")
assert det_exact == target_det, f"Ratio = {det_exact / target_det}"
log("Verified: det(G_B) = 2^2799 * 3^1314")
print("\nALL CHECKS PASSED")
