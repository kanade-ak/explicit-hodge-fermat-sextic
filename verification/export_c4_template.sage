"""
Export the sparse Griffiths residue template detC4 for Z_DEC20^C4 (CI(1,2,2), deg 4)
modulo q1=10009, q2=10333 into data/cycle_templates_mod_q_v2.json.
"""
import itertools
import json
import os
import time

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.2f}s] {msg}", flush=True)

K0 = CyclotomicField(12, 'z')
z0 = K0.gen()
PK = PolynomialRing(K0, 't')
tK = PK.gen()
f_irr = list((tK**6 - 108).factor())[0][0]
K_rel = K0.extension(f_irr, 'gam')
K = K_rel.absolute_field('a')
to_abs = K_rel.embeddings(K)[0]
gam = to_abs(K_rel.gen())
z = to_abs(K_rel(z0))
I_u = z**3
beta = -(gam**2) / K(3)
a_c = beta / K(2)
assert a_c**3 == K(-1) / K(2)

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
    assert sum(g*h for g, h in zip(gs, hs)) == f_fermat
    M = matrix(R, [[p.derivative(v) for v in xs] for p in gs + hs])
    return red(M.determinant())

def pair(d1, d2):
    tot = K(0)
    for b, c1 in d1.items():
        bb = tuple(4 - u for u in b)
        if bb in d2:
            tot += c1 * d2[bb]
    return tot

C_norm = K(-1) / K(279936)

P1 = x0**2 + I_u * x0 * x1 + x1**2
P2 = x0**2 - I_u * x0 * x1 + x1**2
g1 = x2**2 - a_c * P1
g2 = x3**2 - a_c * P2
g3 = x4 - z * x5
h1 = x2**4 + x2**2 * (a_c * P1) + (a_c * P1)**2
h2 = x3**4 + x3**2 * (a_c * P2) + (a_c * P2)**2
h3, rem = (x4**6 + x5**6).quo_rem(g3)
assert rem == 0
detC4 = ci_det([g1, g2, g3], [h1, h2, h3])
prim_sq = C_norm * pair(detC4, detC4)
log(f"Z_DEC20^C4 (CI(1,2,2), deg 4): [Z]_prim^2 = {prim_sq}, [Z]^2 = {prim_sq + QQ(16)/QQ(6)}")
assert prim_sq == K(QQ(160)/QQ(3)), prim_sq

path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "data", "cycle_templates_mod_q.json")
with open(path, "r", encoding="utf-8") as fp:
    data = json.load(fp)

f_min = K.defining_polynomial()
for pdata in data["primes"]:
    q = pdata["q"]
    Fq = GF(q)
    # Match the root of K mod q reproducing z6 and detP
    z6_target = Fq(pdata["z6"])
    detP_ref = {tuple(e): Fq(c) for e, c in pdata["detP"]}
    gP = [x0 - z*x1, x2 - z*x3, x4 - z*x5]
    hP = [(x0**6 + x1**6).quo_rem(gP[0])[0], (x2**6 + x3**6).quo_rem(gP[1])[0], (x4**6 + x5**6).quo_rem(gP[2])[0]]
    detP = ci_det(gP, hP)
    matched_rt = None
    for rt, _ in f_min.change_ring(Fq).roots():
        def to_Fq(c):
            cs = c.list()
            return sum(Fq(cs[k]) * (rt**k) for k in range(len(cs)))
        if to_Fq(z**2) != z6_target:
            continue
        ok = all(to_Fq(detP[b]) == detP_ref[b] for b in detP_ref)
        if ok:
            matched_rt = rt
            break
    assert matched_rt is not None, f"Could not match root mod q={q}"
    def to_Fq_m(c):
        cs = c.list()
        return sum(Fq(cs[k]) * (matched_rt**k) for k in range(len(cs)))
    pdata["detC4"] = [[list(b), int(to_Fq_m(c))] for b, c in sorted(detC4.items())]
    log(f"q={q}: matched_rt={matched_rt}, detC4 exported ({len(pdata['detC4'])} terms)")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "data", "cycle_templates_mod_q_v2.json")
with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(data, fp)
log(f"Saved: {out_path}")
print("\nALL CHECKS PASSED")
