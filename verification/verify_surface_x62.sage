"""
Self-contained exact verification for Theorem A, Proposition 3.1(1), and Table 1
on the Fermat sextic surface X_6^2 over Z (no floating-point arithmetic).
"""
import itertools
import time
from collections import Counter

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.2f}s] {msg}", flush=True)

log("=== Theorem A & Table 1 Verification: Integral Generation of NS(X_6^2) ===")

d = 6; m = d - 1
tuples3 = list(itertools.product(range(m), repeat=3))
F1 = matrix(ZZ, m, m, lambda i, j: 1 if i == j else (-1 if j == i + 1 else 0))
F = F1.tensor_product(F1).tensor_product(F1)
Psi = -(F + F.T)
assert Psi.rank() == 105

K6 = CyclotomicField(6, 'w')
w = K6.gen()

def period_rows(chars):
    rows = []
    for a in chars:
        v0, v1 = [], []
        for b in tuples3:
            c0, c1 = (w**(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])).list()
            v0.append(ZZ(c0)); v1.append(ZZ(c1))
        rows.extend([v0, v1])
    return rows

nonhdg_chars = [a for a in itertools.product(range(1, d), repeat=4) if sum(a) == d]
nl_chars = [a for a in itertools.product(range(1, d), repeat=4)
            if sum(a) == 2*d and sorted(a) in ([1, 3, 4, 4], [2, 2, 3, 5])]
lin_chars = [a for a in itertools.product(range(1, d), repeat=4)
             if sum(a) == 2*d and sorted(a) not in ([1, 3, 4, 4], [2, 2, 3, 5])]
assert len(nonhdg_chars) == 10 and len(nl_chars) == 24 and len(lin_chars) == 61

def pham_sublattice(ann_chars):
    B = matrix(ZZ, period_rows(ann_chars)).right_kernel().basis_matrix()
    G = B * Psi * B.T
    D = G.smith_form()[0]
    snf = [int(D[i, i]) for i in range(D.nrows()) if D[i, i] != 0]
    return len(snf), prod(snf), sorted(Counter(snf).items())

rk_NS0, det_NS0, snf_NS0 = pham_sublattice(nonhdg_chars)
rk_V0, det_V0, snf_V0 = pham_sublattice(nonhdg_chars + nl_chars)
rk_WNL, det_WNL, snf_WNL = pham_sublattice(nonhdg_chars + lin_chars)
assert rk_NS0 == 85 and det_NS0 == (2**41) * (3**31)
assert rk_V0 == 61 and det_V0 == (2**25) * (3**31)
assert rk_WNL == 24 and det_WNL == (2**16) * (3**24)
glue_sq = (det_V0 * det_WNL) // det_NS0
assert glue_sq == 3**24
log(f"1. Pham lattices verified:")
log(f"   NS(X_6^2)_0 : rk={rk_NS0}, |det|={factor(det_NS0)}, SNF={snf_NS0}")
log(f"   V_0(X_6^2)  : rk={rk_V0}, |det|={factor(det_V0)}, SNF={snf_V0}")
log(f"   W_NL(X_6^2) : rk={rk_WNL}, |det|={factor(det_WNL)}, SNF={snf_WNL}")
log(f"   Glue |NS_0 / (V_0 + W_NL)| = 3^12")

K0 = CyclotomicField(12, 'z')
PK = PolynomialRing(K0, 't')
K_rel = K0.extension(PK.gen()**3 - 2, 'c')
K = K_rel.absolute_field('u')
assert K.degree() == 12
to_abs = K_rel.embeddings(K)[0]
z = to_abs(K_rel(K0.gen()))
I_u = z**3
cbrt2 = to_abs(K_rel.gen())
a_c = -K(1) / cbrt2
beta_c = K(2) * a_c
assert z**6 == -1 and I_u**2 == -1 and cbrt2**3 == 2 and a_c**3 == K(-1)/K(2)
log(f"2. Minimal defining field K = Q(zeta_12, 2^(1/3)) verified: [K:Q] = {K.degree()}")

R2 = PolynomialRing(K, 'x0,x1,x2,x3')
x0, x1, x2, x3 = R2.gens(); xs = [x0, x1, x2, x3]
f2 = sum(x**6 for x in xs)

def check_smooth_curve(name, gs):
    I = R2.ideal(gs)
    assert f2 in I, f"{name}: f2 not in ideal!"
    pdim = I.dimension() - 1
    assert pdim == 1, f"{name}: expected proj dim 1, got {pdim}"
    J = matrix(R2, [[g.derivative(v) for v in xs] for g in gs])
    sing_dim = R2.ideal(list(gs) + list(J.minors(2))).dimension() - 1
    assert sing_dim <= -1, f"{name}: not smooth (sing proj dim = {sing_dim})"
    log(f"   [Prop 3.1(1)] {name:6s}: proj_dim={pdim}, f2 in I=True, sing_proj_dim={sing_dim} (SMOOTH)")

gL = [x0 - z*x1, x2 - z*x3]
P1 = x0**2 + I_u*x0*x1 + x1**2; P2 = x0**2 - I_u*x0*x1 + x1**2
gC4 = [x2**2 - a_c*P1, x3**2 - a_c*P2]
gCc = [x3**2 - cbrt2*x0*x2, x1**3 - I_u*(x0**3 + x2**3)]

check_smooth_curve("L", gL)
check_smooth_curve("C_4", gC4)
check_smooth_curve("C_cube", gCc)

def red2(poly):
    out = {}
    for mon, c in zip(poly.monomials(), poly.coefficients()):
        e = tuple(mon.exponents()[0])
        if all(u <= 4 for u in e):
            out[e] = out.get(e, K(0)) + c
    return {e: c for e, c in out.items() if c != 0}

def ci_det2(gs, hs):
    assert sum(g*h for g, h in zip(gs, hs)) == f2
    M = matrix(R2, [[p.derivative(v) for v in xs] for p in gs + hs])
    return red2(M.determinant())

def pair2(d1, d2):
    return sum((c1 * d2[tuple(4-u for u in b)] for b, c1 in d1.items() if tuple(4-u for u in b) in d2), K(0))

hL = [(x0**6 + x1**6).quo_rem(gL[0])[0], (x2**6 + x3**6).quo_rem(gL[1])[0]]
detL = ci_det2(gL, hL)
C2 = K(QQ(-25)/QQ(6)) / pair2(detL, detL)
assert C2 == K(-1) / K(7776)

hC4 = [x2**4 + x2**2*(a_c*P1) + (a_c*P1)**2, x3**4 + x3**2*(a_c*P2) + (a_c*P2)**2]
detC4 = ci_det2(gC4, hC4)
assert C2 * pair2(detC4, detC4) == K(QQ(-32)/QQ(3))

hCc = [x3**4 + x3**2*(cbrt2*x0*x2) + (cbrt2*x0*x2)**2, x1**3 + I_u*(x0**3 + x2**3)]
detCc = ci_det2(gCc, hCc)
assert C2 * pair2(detCc, detCc) == K(-12)

# Factor out gam (gam^2 = -3*beta_c) so the determinant is defined over K
q0 = x3**2 - a_c * (x0**2)
gA1 = x2**2 + x3**2 - beta_c * (x0**2)
hA1, remA = (x0**6 + x2**6 + x3**6 - K(3) * beta_c * (x0**2) * (q0**2)).quo_rem(gA1)
assert remA == 0
M_A0 = matrix(R2, [[p.derivative(v) for v in xs] for p in [gA1, x1**3, hA1, x0*q0]])
detA_over_gam = red2(K(2) * M_A0.determinant())
assert C2 * (-K(3) * beta_c) * pair2(detA_over_gam, detA_over_gam) == K(-12)
log("3. Exact Griffiths residue determinants verified: L^2=-4, C_Aoki^2=-6, C_4^2=-8, C_cube^2=-6")

B85 = sorted([b for b in itertools.product(range(5), repeat=4) if sum(b) == 8])
idx85 = {b: i for i, b in enumerate(B85)}
conj85 = [idx85[tuple(4 - u for u in b)] for b in B85]

def materialize_mod_q(q):
    Fq = GF(q)
    rt = K.defining_polynomial().change_ring(Fq).roots()[0][0]
    def to_Fq(c):
        cs = c.list()
        return sum(Fq(cs[k]) * (rt**k) for k in range(len(cs)))
    z6p = [to_Fq(z**2)**k for k in range(6)]
    C2_q = to_Fq(C2)
    gam_q = to_Fq(-K(3) * beta_c).sqrt()
    assert gam_q**2 == to_Fq(-K(3) * beta_c)
    def mk_vec(d_Fq, perm, sh):
        inv = sum(1 for i in range(4) for j in range(i+1, 4) if perm[i] > perm[j])
        sgn = Fq(1) if inv % 2 == 0 else Fq(-1)
        v = [Fq(0)] * 85
        for b, c in d_Fq.items():
            nb = [0]*4
            for k in range(4): nb[perm[k]] = b[k]
            nb = tuple(nb)
            pw = sum(sh[k] * (nb[k] + 1) for k in range(4)) % 6
            v[idx85[nb]] += c * sgn * z6p[pw]
        return v
    dL = {b: to_Fq(c) for b, c in detL.items()}
    dA = {b: gam_q * to_Fq(c) for b, c in detA_over_gam.items()}
    dC4 = {b: to_Fq(c) for b, c in detC4.items()}
    dCc = {b: to_Fq(c) for b, c in detCc.items()}
    L0 = mk_vec(dL, (0,1,2,3), (0,0,0,0))
    vecs_L = []
    for perm in [(0,1,2,3), (0,2,1,3), (0,3,1,2)]:
        for a0, a1 in itertools.product(range(6), repeat=2):
            sh = [0]*4; sh[perm[1]] = a0; sh[perm[3]] = a1
            v = mk_vec(dL, perm, sh)
            vecs_L.append([v[i] - L0[i] for i in range(85)])
    vecs_L = vecs_L[1:]
    vecs_A = []
    for perm in itertools.permutations(range(4)):
        if perm[2] > perm[3]: continue
        for s0, s1, s2 in itertools.product(range(6), range(2), range(3)):
            sh = [0]*4; sh[perm[0]] = s0; sh[perm[1]] = s1; sh[perm[2]] = s2
            vecs_A.append(mk_vec(dA, perm, sh))
    vecs_C4 = []
    for perm in itertools.permutations(range(4)):
        if perm[0] > perm[1] or perm[2] > perm[3]: continue
        for s0, s1, s2 in itertools.product(range(6), range(6), range(3)):
            sh = [0]*4; sh[perm[0]] = s0; sh[perm[1]] = s1; sh[perm[2]] = s2
            v = mk_vec(dC4, perm, sh)
            vecs_C4.append([v[i] - Fq(4) * L0[i] for i in range(85)])
    vecs_Cc = []
    for perm in itertools.permutations(range(4)):
        if perm[0] > perm[2]: continue
        for s0, s1, s2 in itertools.product(range(6), range(2), range(6)):
            sh = [0]*4; sh[perm[0]] = s0; sh[perm[1]] = s1; sh[perm[2]] = s2
            vecs_Cc.append(mk_vec(dCc, perm, sh))
    return Fq, C2_q, vecs_L, vecs_A, vecs_C4, vecs_Cc

Fq1, C2_1, vL1, vA1, vC4_1, vCc_1 = materialize_mod_q(10009)
Fq2, C2_2, vL2, vA2, vC4_2, vCc_2 = materialize_mod_q(10333)

def compute_exact_index_and_snf(name, l1, l2, expected_rk, target_det):
    M1 = matrix(Fq1, l1)
    M2 = matrix(Fq2, l2)
    assert M1.rank() == expected_rk and M2.rank() == expected_rk
    pivs = M1.transpose().pivots()
    B1 = M1.matrix_from_rows(pivs)
    B2 = M2.matrix_from_rows(pivs)
    B1c = matrix(Fq1, [[B1[r, conj85[c]] for c in range(85)] for r in range(expected_rk)])
    B2c = matrix(Fq2, [[B2[r, conj85[c]] for c in range(85)] for r in range(expected_rk)])
    def rz(mat_q, q_):
        half = q_ // 2
        return matrix(ZZ, [[int(mat_q[i,j]) - q_ if int(mat_q[i,j]) > half else int(mat_q[i,j])
                            for j in range(mat_q.ncols())] for i in range(mat_q.nrows())])
    GB1 = rz(C2_1 * (B1 * B1c.T), 10009)
    GB2 = rz(C2_2 * (B2 * B2c.T), 10333)
    assert GB1 == GB2, "q1 vs q2 mismatch on G_B!"
    H1 = rz(C2_1 * (M1 * B1c.T), 10009)
    H2 = rz(C2_2 * (M2 * B2c.T), 10333)
    assert H1 == H2, "q1 vs q2 mismatch on H!"
    # HNF(H1)[:r] = T * GB1 gives Z-basis T of Lambda and Gram matrix R_H * GB1^{-1} * R_H^T
    R_H = H1.hermite_form()[:expected_rk, :]
    det_H = prod(abs(R_H[i, i]) for i in range(expected_rk))
    det_GB = abs(GB1.det())
    idx = ZZ(sqrt(QQ(det_H**2) / QQ(det_GB * target_det)))
    G_Lam_Q = R_H.change_ring(QQ) * GB1.change_ring(QQ).inverse() * R_H.change_ring(QQ).transpose()
    G_Lam = matrix(ZZ, [[ZZ(G_Lam_Q[i, j]) for j in range(expected_rk)] for i in range(expected_rk)])
    D_Lam = G_Lam.smith_form()[0]
    snf_Lam = sorted(Counter(int(D_Lam[i, i]) for i in range(expected_rk)).items())
    log(f"4. [{name}] (rk={expected_rk}, q1==q2 on {H1.nrows()}x{expected_rk}): index = {idx} ({factor(idx) if idx > 1 else 'INDEX 1'})")
    log(f"      SNF = {snf_Lam}")
    return idx, snf_Lam

idx_L, snf_L = compute_exact_index_and_snf("108 lines (in V_0)", vL1, vL2, 61, det_V0)
assert idx_L == 1 and snf_L == snf_V0

idx_A, snf_A = compute_exact_index_and_snf("<lines, C_Aoki>", vL1 + vA1, vL2 + vA2, 85, det_NS0)
assert idx_A == (2**16) * (3**6)
assert snf_A == [(1, 48), (2, 1), (4, 4), (12, 22), (36, 9), (108, 1)]

idx_C4, snf_C4 = compute_exact_index_and_snf("<lines, C_4>", vL1 + vC4_1, vL2 + vC4_2, 85, det_NS0)
assert idx_C4 == 2**16
assert snf_C4 == [(1, 48), (2, 1), (4, 16), (12, 10), (36, 9), (108, 1)]

idx_full, snf_full = compute_exact_index_and_snf("<lines, C_4, C_cube>", vL1 + vC4_1 + vCc_1, vL2 + vC4_2 + vCc_2, 85, det_NS0)
assert idx_full == 1 and snf_full == snf_NS0

log("THEOREM A, PROPOSITION 3.1(1), AND ALL 4 ROWS OF TABLE 1 VERIFIED OVER Z!")
print("\nALL CHECKS PASSED")
