"""
Export sparse Griffiths residue templates for P, Z_DEC20^Aoki, Z_J11^Aoki,
and the symmetric CI(2,3,1) cycle modulo dual split primes q1=10009, q2=10333
to data/cycle_templates_mod_q.json (Lemma 2.1: 6 * |<Z1, Z2>| <= 576 < q1/2).
"""
import json
import os
import time

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.2f}s] {msg}", flush=True)

K0 = CyclotomicField(12, 'z')
z0 = K0.gen()
PK = PolynomialRing(K0, 't')
t = PK.gen()
f_irr = list((t**6 - 108).factor())[0][0]
K_rel = K0.extension(f_irr, 'gam')
K = K_rel.absolute_field('a')
to_abs = K_rel.embeddings(K)[0]
gam = to_abs(K_rel.gen())
z = to_abs(K_rel(z0))
I_unit = z**3
beta = -(gam**2) / K(3)
C_norm = K(-1) / K(279936)

R = PolynomialRing(K, 'x0,x1,x2,x3,x4,x5')
x0, x1, x2, x3, x4, x5 = R.gens()
xs = [x0, x1, x2, x3, x4, x5]

def reduce_fermat_jac(poly):
    out = {}
    for mon, coeff in zip(poly.monomials(), poly.coefficients()):
        e = tuple(mon.exponents()[0])
        if all(u <= 4 for u in e):
            out[e] = out.get(e, K(0)) + coeff
    return {e: c for e, c in out.items() if c != 0}

def complete_intersection_det(gs, hs):
    rows = [[p.derivative(v) for v in xs] for p in list(gs) + list(hs)]
    return reduce_fermat_jac(matrix(R, rows).determinant())

gP = [x0 - z*x1, x2 - z*x3, x4 - z*x5]
hP = [(x0**6 + x1**6).quo_rem(gP[0])[0],
      (x2**6 + x3**6).quo_rem(gP[1])[0],
      (x4**6 + x5**6).quo_rem(gP[2])[0]]
detP = complete_intersection_det(gP, hP)

q0 = x3**2 - (beta / K(2)) * (x0**2)
gD1 = x2**2 + x3**2 - beta * (x0**2)
hD1, _ = (x0**6 + x2**6 + x3**6 - 3 * beta * (x0**2) * (q0**2)).quo_rem(gD1)
gD2 = x1**3 - gam * x0 * q0
hD2 = x1**3 + gam * x0 * q0
gD3 = x4 - z * x5
hD3, _ = (x4**6 + x5**6).quo_rem(gD3)
detD = complete_intersection_det([gD1, gD2, gD3], [hD1, hD2, hD3])

q1_p = x5**2 - (beta / K(2)) * (x1**2)
gJ2 = x4**2 + x5**2 - beta * (x1**2)
hJ2, _ = (x1**6 + x4**6 + x5**6 - 3 * beta * (x1**2) * (q1_p**2)).quo_rem(gJ2)
gJ3 = x0 * q0 - I_unit * x1 * q1_p
hJ3 = 3 * beta * (x0 * q0 + I_unit * x1 * q1_p)
detJ = complete_intersection_det([gD1, gJ2, gJ3], [hD1, hJ2, hJ3])

SQRT_M3 = 2 * (z**2) - 1
gS1 = x0**2 + x2**2 + x3**2
hS1, _ = (x0**6 + x2**6 + x3**6 - 3 * (x0**2) * (x2**2) * (x3**2)).quo_rem(gS1)
detSymD = complete_intersection_det([gS1, x1**3 - SQRT_M3*x0*x2*x3, gD3],
                                    [hS1, x1**3 + SQRT_M3*x0*x2*x3, hD3])

f_abs = K.defining_polynomial()
split_primes = []
p = next_prime(10000)
while len(split_primes) < 2:
    rts = f_abs.change_ring(GF(p)).roots()
    if len(rts) == K.degree():
        split_primes.append((int(p), int(rts[0][0])))
    p = next_prime(p)
log(f"Found 2 split primes: {split_primes}")

def monoms_12():
    res = []
    def rec(i, rem, cur):
        if i == 5:
            if 0 <= rem <= 4:
                res.append(tuple(cur + [rem]))
            return
        for v in range(max(0, rem - 4*(5-i)), min(4, rem) + 1):
            rec(i+1, rem-v, cur+[v])
    rec(0, 12, [])
    return res

B12 = monoms_12()
assert len(B12) == 1751

def export_for_prime(q, r_a):
    Fq = GF(q)
    def to_fq(c):
        coeffs = c.list()
        return int(sum(Fq(coeffs[k]) * (Fq(r_a) ** k) for k in range(len(coeffs))))
    z6_q = to_fq(z**2)
    assert pow(z6_q, 6, q) == 1 and pow(z6_q, 3, q) == q - 1
    cnorm_q = int(Fq(-1) / Fq(279936))
    def enc(d):
        return [[list(b), to_fq(c)] for b, c in sorted(d.items())]
    return {
        "q": q,
        "z6": z6_q,
        "C_norm_mod_q": cnorm_q,
        "detP": enc(detP),
        "detD_Aoki": enc(detD),
        "detJ_Aoki": enc(detJ),
        "detSymD": enc(detSymD),
    }

payload = {
    "B12": [list(b) for b in B12],
    "primes": [export_for_prime(q, r_a) for q, r_a in split_primes],
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "data", "cycle_templates_mod_q.json")
with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(payload, fp, default=int)
log(f"Exported: {out_path}")
print("ALL CHECKS PASSED")
