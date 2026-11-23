"""
Verification of Proposition 3.1(2) for the five surface families on X_6^4:
checks projective dimension 2, containment in X_6^4, and singular locus
dimension <= 0 (normal reduced via R_1 + S_2) over K = Q(zeta_12, 2^(1/3)) and F_10009.
"""
import time
T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.2f}s] {msg}", flush=True)

def verify_over_field(label, F, z, I_u, a_c, beta, cbrt2):
    log(f"--- Checking all 5 surfaces over {label} ---")
    R = PolynomialRing(F, 'x0,x1,x2,x3,x4,x5')
    x0, x1, x2, x3, x4, x5 = R.gens()
    xs = [x0, x1, x2, x3, x4, x5]
    f_fermat = sum(x**6 for x in xs)

    surfaces = [
        ("1. P (CI(1,1,1), deg 1)",
         [x0 - z*x1, x2 - z*x3, x4 - z*x5]),
        ("2. Z_DEC20^C4 (CI(1,2,2), deg 4)",
         [x2**2 - a_c*(x0**2 + I_u*x0*x1 + x1**2),
          x3**2 - a_c*(x0**2 - I_u*x0*x1 + x1**2),
          x4 - z*x5]),
        ("3. Z_J11^Aoki (CI(2,2,3), deg 12)",
         [x2**2 + x3**2 - beta*x0**2,
          x4**2 + x5**2 - beta*x1**2,
          x0*(x3**2 - a_c*x0**2) - I_u*x1*(x5**2 - a_c*x1**2)]),
        ("4. Z_DEC20^cube (CI(1,2,3), deg 6)",
         [x3**2 - cbrt2*x0*x2,
          x1**3 - I_u*(x0**3 + x2**3),
          x4 - z*x5]),
        ("5. Z_J11^cube (CI(2,2,3), deg 12)",
         [x3**2 - cbrt2*x0*x2,
          x5**2 - cbrt2*x1*x4,
          (x0**3 + x2**3) - I_u*(x1**3 + x4**3)]),
    ]

    for name, gs in surfaces:
        I_Z = R.ideal(gs)
        dim_proj = I_Z.dimension() - 1
        assert dim_proj == 2, f"{name}: dim_proj = {dim_proj} != 2!"
        assert f_fermat in I_Z, f"{name}: X_6^4 does not contain Z!"
        J = matrix(R, [[g.derivative(v) for v in xs] for g in gs])
        minors3 = J.minors(3)
        I_sing = R.ideal(gs + minors3)
        sing_dim_proj = I_sing.dimension() - 1
        assert sing_dim_proj <= 0, f"{name}: singular locus has proj dim {sing_dim_proj} > 0!"
        status = (
            "smooth (empty sing locus)"
            if sing_dim_proj < 0
            else f"isolated singularities (sing_proj_dim = {sing_dim_proj}, normal CI via R_1+S_2)"
        )
        log(f"  {name:36s}: proj_dim={dim_proj}, f_Fermat in I_Z=True, {status}")

K0 = CyclotomicField(12, 'z')
PK = PolynomialRing(K0, 't')
K_rel = K0.extension(PK.gen()**3 - 2, 'c')
K = K_rel.absolute_field('u')
assert K.degree() == 12
to_abs = K_rel.embeddings(K)[0]
z_K = to_abs(K_rel(K0.gen()))
I_K = z_K**3
cbrt2_K = to_abs(K_rel.gen())
a_K = -K(1) / cbrt2_K
beta_K = K(2) * a_K
assert z_K**6 == -1 and I_K**2 == -1 and cbrt2_K**3 == 2 and beta_K**3 == -4
verify_over_field("characteristic 0: K = Q(zeta_12, 2^(1/3))", K, z_K, I_K, a_K, beta_K, cbrt2_K)

q = 10009
Fq = GF(q)
R1 = PolynomialRing(Fq, 'u'); u = R1.gen()
z_q = [r for r, _ in (u**12 - 1).roots() if r**6 == -1 and r**2 != -1][0]
I_q = z_q**3
a_q = [r for r, _ in (u**3 + Fq(1)/Fq(2)).roots()][0]
beta_q = 2 * a_q
cbrt2_q = -Fq(1) / a_q
assert beta_q**3 == -4 and cbrt2_q**3 == 2
verify_over_field(f"good reduction F_{q}", Fq, z_q, I_q, a_q, beta_q, cbrt2_q)

print("\nALL CHECKS PASSED")
