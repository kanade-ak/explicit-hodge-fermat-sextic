"""
Verification for Section 6.3(1): evaluates Z_DEC20^C4 (CI(1,2,2), deg 4) through the
exact Pham-coordinate bridge of Santana de Jesus (2026, Lemma 2.3), checking integrality
of 2*chat(a) in Z[zeta_6], integrality of v = Psi*c in Z^3125 for [Z_DEC20^C4] - 4[P_0],
and exact pairing <Z_DEC20^C4, P_0>.
"""
import itertools
import time
import lzma
import numpy as np
from sage.all import CyclotomicField, PolynomialRing, QQ, ZZ, matrix

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.2f}s] {msg}", flush=True)

# Setup from Santana de Jesus (2026), Lemma 2.3
K0 = CyclotomicField(12)
eta = K0.gen()
z6 = eta ** 2
L = K0.extension(PolynomialRing(K0, "t").gen() ** 3 - 2, "lam")
lam = L.gen()
I_ = L(eta ** 3)
sqrt3 = L(eta + eta ** 11)
a_c = -L(1) / lam
beta = 2 * a_c
assert a_c**3 == L(-1)/L(2) and beta**3 == -4

Rx = PolynomialRing(L, "x0,x1,x2,x3,x4,x5")
x = Rx.gens()
F6 = sum(v ** 6 for v in x)
MONS = [m for m in itertools.product(range(5), repeat=6) if sum(m) == 12]
NH = len(MONS)
mon_index = {m: i for i, m in enumerate(MONS)}
A6 = [tuple(v + 1 for v in m) for m in MONS]
pos = np.array([sum((a[k] - 1) * 5 ** (5 - k) for k in range(1, 6)) for a in A6])
neg = np.array([mon_index[tuple(4 - v for v in m)] for m in MONS])

def alpha(a):
    n = [sum(1 for v in a if v == j) for j in range(6)]
    q, r = divmod(-n[1] + 3 * n[4] + 4 * n[5], 3)
    e3 = n[1] - n[4] - n[5]
    s = {1: L(1) / 2, 2: sqrt3 / 2, 3: L(1), 4: sqrt3 / 2, 5: L(1) / 2}[a[0]]
    return L(2) ** q * lam ** r * L(3) ** (e3 // 2) * (sqrt3 if e3 % 2 else 1) * s * L(eta) ** (sum(a[1:]) % 12)

def prodz(a):
    out = L(1)
    for v in a[1:]:
        out *= L(z6) ** v - 1
    return out

KAPPA = QQ(3) / 2
rho = [1 / (KAPPA * alpha(a) * prodz(a)) for a in A6]
rho_e, rho_u = [], []
for r_ in rho:
    lst = r_.list()
    nz_ = [j for j in range(3) if lst[j] != 0]
    rho_e.append(nz_[0])
    rho_u.append(lst[nz_[0]])

F1 = matrix(K0, 5, 5, [[z6 ** (a * b) for b in range(5)] for a in range(1, 6)])
G = F1.inverse()
V1 = np.eye(5, dtype=np.int64)
for j in range(4):
    V1[j, j + 1] = -1
Vk = V1
for _ in range(4):
    Vk = np.kron(Vk, V1)
Psi = Vk + Vk.T
del Vk

def zpair(k):
    c = K0(k).list()
    assert c[1] == 0 and c[3] == 0
    return c[0], c[2]

Gr = np.array([[int(zpair(6 * G[i, j])[0]) for j in range(5)] for i in range(5)], dtype=np.int64)
Gi = np.array([[int(zpair(6 * G[i, j])[1]) for j in range(5)] for i in range(5)], dtype=np.int64)

def functional(fs, gs):
    assert sum(f * g for f, g in zip(fs, gs)) == F6, "F6 != sum f*g!"
    det = matrix(Rx, [[f.derivative(v) for v in x] for f in list(fs) + list(gs)]).det()
    out = {}
    for mon, c in det.dict().items():
        mon = tuple(mon)
        if max(mon) <= 4:
            out[mon_index[tuple(4 - v for v in mon)]] = c
    return out

def chat2(Phi):
    bad = {"lam": 0, "odd": 0, "den": 0}
    re = np.zeros(NH, dtype=np.int64)
    im = np.zeros(NH, dtype=np.int64)
    for i, c in Phi.items():
        lst = c.list()
        e = rho_e[i]
        j0 = (-e) % 3
        if any(lst[j] != 0 for j in range(3) if j != j0):
            bad["lam"] += 1
            continue
        cc = (rho_u[i] * lst[j0] * (2 if j0 + e == 3 else 1)).list()
        if cc[1] != 0 or cc[3] != 0:
            bad["odd"] += 1
            continue
        r0, r2 = 2 * QQ(cc[0]), 2 * QQ(cc[2])
        if r0.denominator() != 1 or r2.denominator() != 1:
            bad["den"] += 1
            continue
        re[i], im[i] = int(r0), int(r2)
    return re, im, bad

def pham_pairing_from_chat2(re, im):
    """Compute v = Psi @ c via (6*G)^{tensor 5} over Z[zeta_6], scaled by 2 * 6^5."""
    T_r = np.zeros(3125, dtype=np.int64)
    T_i = np.zeros(3125, dtype=np.int64)
    T_r[pos] = re
    T_i[pos] = im
    A_r = T_r.reshape((5, 5, 5, 5, 5)).astype(object)
    A_i = T_i.reshape((5, 5, 5, 5, 5)).astype(object)
    Gr_o = Gr.astype(object); Gi_o = Gi.astype(object)
    for ax in range(5):
        A_r = np.moveaxis(A_r, ax, 0).reshape(5, -1)
        A_i = np.moveaxis(A_i, ax, 0).reshape(5, -1)
        nr = Gr_o @ A_r - Gi_o @ A_i
        ni = Gr_o @ A_i + Gi_o @ A_r + Gi_o @ A_i
        A_r = np.moveaxis(nr.reshape(5, 5, 5, 5, 5), 0, ax)
        A_i = np.moveaxis(ni.reshape(5, 5, 5, 5, 5), 0, ax)
    c_r = A_r.reshape(3125)
    c_i = A_i.reshape(3125)
    assert np.all(c_i == 0), f"Non-zero imaginary part in Pham coords! max={np.max(np.abs(c_i))}"
    v_scaled = Psi.astype(object) @ c_r
    den = 2 * (6 ** 5)
    rems = np.array([int(val % den) for val in v_scaled])
    nz_rems = np.count_nonzero(rems)
    return v_scaled, den, nz_rems

x0, x1, x2, x3, x4, x5 = x
def geom6(u, v, e):
    return sum(u**(5 - k) * ((e * v)**k) for k in range(6))

fP = [x0 - eta*x1, x2 - eta*x3, x4 - eta*x5]
gP = [geom6(x0, x1, eta), geom6(x2, x3, eta), geom6(x4, x5, eta)]
Phi_P = functional(fP, gP)
re_P, im_P, bad_P = chat2(Phi_P)
log(f"Plane P0: bad={bad_P}")
vP_s, den, nzP = pham_pairing_from_chat2(re_P, im_P)
log(f"  P0: v_scaled mod {den} non-zero = {nzP}/3125; 6*P0 mod {den} non-zero = {np.count_nonzero([int((6*val)%den) for val in vP_s])}")

P1 = x0**2 + I_*x0*x1 + x1**2
P2 = x0**2 - I_*x0*x1 + x1**2
fC4 = [x2**2 - a_c*P1, x3**2 - a_c*P2, x4 - eta*x5]
gC4 = [x2**4 + x2**2*(a_c*P1) + (a_c*P1)**2,
       x3**4 + x3**2*(a_c*P2) + (a_c*P2)**2,
       geom6(x4, x5, eta)]
Phi_C4 = functional(fC4, gC4)
re_C4, im_C4, bad_C4 = chat2(Phi_C4)
log(f"\nZ_DEC20^C4 (CI(1,2,2), deg 4): bad={bad_C4}")
vC4_s, _, nzC4 = pham_pairing_from_chat2(re_C4, im_C4)
nz_minus = np.count_nonzero([int((vC4_s[k] - 4*vP_s[k]) % den) for k in range(3125)])
nz_plus  = np.count_nonzero([int((vC4_s[k] + 4*vP_s[k]) % den) for k in range(3125)])
log(f"  [Z_C4] - 4[P0] non-zero mod {den} = {nz_minus}/3125")
log(f"  [Z_C4] + 4[P0] non-zero mod {den} = {nz_plus}/3125")

for mult in [1, 2, 3, 6]:
    nm = np.count_nonzero([int((mult * (vC4_s[k] - 4*vP_s[k])) % den) for k in range(3125)])
    np_ = np.count_nonzero([int((mult * (vC4_s[k] + 4*vP_s[k])) % den) for k in range(3125)])
    log(f"    mult={mult}: minus non-zero={nm}, plus non-zero={np_}")

def res_pair(p1, p2):
    s = L(0)
    for i, c in p1.items():
        j = int(neg[i])
        if j in p2:
            s += c * p2[j]
    return -s / 6**7

log(f"\n  <C4, C4>_prim = {res_pair(Phi_C4, Phi_C4)} (expected 160/3)")
log(f"  <C4, P0>_prim = {res_pair(Phi_C4, Phi_P)}")
log(f"    (+1)*<C4, P0>_prim + 4/6 = {res_pair(Phi_C4, Phi_P) + QQ(4)/6}")
log(f"    (-1)*<C4, P0>_prim + 4/6 = {-res_pair(Phi_C4, Phi_P) + QQ(4)/6}")
