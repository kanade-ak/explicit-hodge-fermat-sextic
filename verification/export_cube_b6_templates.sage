"""
Export sparse Griffiths residue templates for Z_DEC20^cube (CI(1,2,3), deg 6),
Z_J11^cube (CI(2,2,3), deg 12), and the B_6 conic-lift templates mod q1=10009,
q2=10333 into data/cycle_templates_mod_q_v3.json.
"""
import itertools
import json
import os
import time
from collections import Counter

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.2f}s] {msg}", flush=True)

K0 = CyclotomicField(12, 'z')
z0 = K0.gen()
PK = PolynomialRing(K0, 't')
f_irr = list((PK.gen()**6 - 108).factor())[0][0]
K_rel = K0.extension(f_irr, 'gam')
K = K_rel.absolute_field('a')
to_abs = K_rel.embeddings(K)[0]
gam = to_abs(K_rel.gen())
z = to_abs(K_rel(z0))
I_u = z**3
beta = -(gam**2) / K(3)
a_c = beta / K(2)
cbrt2 = -K(1) / a_c
assert cbrt2**3 == K(2)

R = PolynomialRing(K, 'x0,x1,x2,x3,x4,x5')
x0, x1, x2, x3, x4, x5 = R.gens()
xs = [x0, x1, x2, x3, x4, x5]
f_fermat = sum(x**6 for x in xs)

def red(poly):
    out = {}
    for mon, c in zip(poly.monomials(), poly.coefficients()):
        e = tuple(mon.exponents()[0])
        if all(u <= 4 for u in e):
            out[e] = out.get(e, K(0)) + c
    return {e: c for e, c in out.items() if c != 0}

def ci_det(gs, hs):
    assert sum(g*h for g, h in zip(gs, hs)) == f_fermat, "f_fermat != sum(g*h)!"
    M = matrix(R, [[p.derivative(v) for v in xs] for p in gs + hs])
    return red(M.determinant())

def pair(d1, d2):
    return sum((c1 * d2[tuple(4-u for u in b)] for b, c1 in d1.items() if tuple(4-u for u in b) in d2), K(0))

def classify_beta(b):
    a = tuple(u + 1 for u in b)
    c = Counter(a)
    if all((cnt % 2 == 0 if x == 3 else c.get(6 - x, 0) == cnt) for x, cnt in c.items()):
        return "LINEAR"
    if any((6 - x) in c and (x != 3 or c[x] >= 2) for x in c):
        return "DECOMP_2_0"
    return "JOIN_1_1"

C_norm = K(-1) / K(279936)

# 1. Z_DEC20^cube (CI(2,3,1), deg 6):
gD1 = x3**2 - cbrt2 * x0 * x2
hD1 = x3**4 + x3**2 * (cbrt2 * x0 * x2) + (cbrt2 * x0 * x2)**2
gD2 = x1**3 - I_u * (x0**3 + x2**3)
hD2 = x1**3 + I_u * (x0**3 + x2**3)
gD3 = x4 - z * x5
hD3, rem = (x4**6 + x5**6).quo_rem(gD3)
assert rem == 0
detD_cube = ci_det([gD1, gD2, gD3], [hD1, hD2, hD3])
pDD = C_norm * pair(detD_cube, detD_cube)
laysD = Counter(classify_beta(b) for b in detD_cube)
log(f"Z_DEC20^cube (CI(2,3,1), deg 6): [Z]_prim^2 = {pDD}, [Z]^2 = {pDD+6}, terms={len(detD_cube)}, layers={dict(laysD)}")
for L_name in ("LINEAR", "DECOMP_2_0", "JOIN_1_1"):
    sub = {b: c for b, c in detD_cube.items() if classify_beta(b) == L_name}
    log(f"    ||π_{L_name}(Z_DEC20^cube)||^2 = {C_norm * pair(sub, sub)}")

# 2. Z_J11^cube (CI(2,2,3), deg 12):
gJ1 = x3**2 - cbrt2 * x0 * x2
hJ1 = x3**4 + x3**2 * (cbrt2 * x0 * x2) + (cbrt2 * x0 * x2)**2
gJ2 = x5**2 - cbrt2 * x1 * x4
hJ2 = x5**4 + x5**2 * (cbrt2 * x1 * x4) + (cbrt2 * x1 * x4)**2
gJ3 = (x0**3 + x2**3) - I_u * (x1**3 + x4**3)
hJ3 = (x0**3 + x2**3) + I_u * (x1**3 + x4**3)
detJ_cube = ci_det([gJ1, gJ2, gJ3], [hJ1, hJ2, hJ3])
pJJ = C_norm * pair(detJ_cube, detJ_cube)
laysJ = Counter(classify_beta(b) for b in detJ_cube)
log(f"Z_J11^cube (CI(2,2,3), deg 12): [Z]_prim^2 = {pJJ}, [Z]^2 = {pJJ+24}, terms={len(detJ_cube)}, layers={dict(laysJ)}")
for L_name in ("LINEAR", "DECOMP_2_0", "JOIN_1_1"):
    sub = {b: c for b, c in detJ_cube.items() if classify_beta(b) == L_name}
    log(f"    ||π_{L_name}(Z_J11^cube)||^2 = {C_norm * pair(sub, sub)}")

# B_6 conic-lift ideal
R_b6 = PolynomialRing(QQ, 'c0,c1,d0,d1,d2,d3')
c0, c1, d0, d1, d2, d3 = R_b6.gens()
Rt = PolynomialRing(R_b6, 't'); tt = Rt.gen()
P2s = (1 + tt**6 + (c0 + c1*tt + c0*tt**2)**3) - (d0 + d1*tt + d2*tt**2 + d3*tt**3)**2
I2s = R_b6.ideal([P2s.coefficient(k) for k in range(7)]).saturation(R_b6.ideal([c1]))[0]

path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "data", "cycle_templates_mod_q_v2.json")
with open(path, "r", encoding="utf-8") as fp:
    data = json.load(fp)

gP = [x0 - z*x1, x2 - z*x3, x4 - z*x5]
hP = [(x0**6 + x1**6).quo_rem(gP[0])[0], (x2**6 + x3**6).quo_rem(gP[1])[0], (x4**6 + x5**6).quo_rem(gP[2])[0]]
detP = ci_det(gP, hP)
f_min = K.defining_polynomial()

for pdata in data["primes"]:
    q = pdata["q"]
    Fq = GF(q)
    z6_target = Fq(pdata["z6"])
    detP_ref = {tuple(e): Fq(c) for e, c in pdata["detP"]}
    matched_rt = None
    for rt, _ in f_min.change_ring(Fq).roots():
        def to_Fq(c):
            cs = c.list()
            return sum(Fq(cs[k]) * (rt**k) for k in range(len(cs)))
        if to_Fq(z**2) != z6_target:
            continue
        if all(to_Fq(detP[b]) == detP_ref[b] for b in detP_ref):
            matched_rt = rt
            break
    assert matched_rt is not None
    def to_Fq_m(c):
        cs = c.list()
        return sum(Fq(cs[k]) * (matched_rt**k) for k in range(len(cs)))
    pdata["detD_cube"] = [[list(b), int(to_Fq_m(c))] for b, c in sorted(detD_cube.items())]
    pdata["detJ_cube"] = [[list(b), int(to_Fq_m(c))] for b, c in sorted(detJ_cube.items())]

    Rq_b = PolynomialRing(Fq, 'c0,c1,d0,d1,d2,d3')
    V_b = Rq_b.ideal([Rq_b(p) for p in I2s.gens()]).variety()
    b6_sols = [s for s in V_b if s['c0'] != 0]
    log(f"q={q}: detD_cube ({len(pdata['detD_cube'])} terms), detJ_cube ({len(pdata['detJ_cube'])} terms), B#6 sols={len(b6_sols)}")
    if b6_sols:
        s = b6_sols[0]
        cc0, cc1 = Fq(s['c0']), Fq(s['c1'])
        dd0, dd1, dd2, dd3 = Fq(s['d0']), Fq(s['d1']), Fq(s['d2']), Fq(s['d3'])
        z_q = to_Fq_m(z)
        I_q = to_Fq_m(I_u)
        Rf = PolynomialRing(Fq, 'x0,x1,x2,x3,x4,x5')
        X0, X1, X2, X3, X4, X5 = Rf.gens(); Xs = [X0, X1, X2, X3, X4, X5]
        Ff = sum(v**6 for v in Xs)
        def ci_det_q(gs, hs):
            assert sum(g*h for g, h in zip(gs, hs)) == Ff
            M = matrix(Rf, [[p.derivative(v) for v in Xs] for p in gs + hs])
            out = {}
            for mon, c in zip(M.determinant().monomials(), M.determinant().coefficients()):
                e = tuple(mon.exponents()[0])
                if all(u <= 4 for u in e):
                    out[e] = out.get(e, Fq(0)) + c
            return {e: c for e, c in out.items() if c != 0}
        Q02 = cc0*X0**2 + cc1*X0*X2 + cc0*X2**2
        G02 = dd0*X0**3 + dd1*X0**2*X2 + dd2*X0*X2**2 + dd3*X2**3
        Q14 = cc0*X1**2 + cc1*X1*X4 + cc0*X4**2
        G14 = dd0*X1**3 + dd1*X1**2*X4 + dd2*X1*X4**2 + dd3*X4**3
        h02 = X3**4 + X3**2*Q02 + Q02**2
        h14 = X5**4 + X5**2*Q14 + Q14**2
        h45 = (X4**6 + X5**6).quo_rem(X4 - z_q*X5)[0]
        dD_b6 = ci_det_q([X3**2 - Q02, X1**3 - I_q*G02, X4 - z_q*X5], [h02, X1**3 + I_q*G02, h45])
        dJ_b6 = ci_det_q([X3**2 - Q02, X5**2 - Q14, G02 - I_q*G14], [h02, h14, G02 + I_q*G14])
        pdata["detD_B6"] = [[list(b), int(c)] for b, c in sorted(dD_b6.items())]
        pdata["detJ_B6"] = [[list(b), int(c)] for b, c in sorted(dJ_b6.items())]
        log(f"    -> exported detD_B6 ({len(dD_b6)} terms), detJ_B6 ({len(dJ_b6)} terms)")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "data", "cycle_templates_mod_q_v3.json")
with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(data, fp)
log(f"Saved: {out_path}")
print("\nALL CHECKS PASSED")
