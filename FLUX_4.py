import sympy as sp

# ---------- Symbols ----------
x, y, z = sp.symbols('x y z', real=True)
beta = sp.symbols('beta', real=True)
g = sp.symbols('g', real=True)  # index/label for "g" direction
alpha, gamma, delta, epsilon = sp.symbols('alpha gamma delta epsilon', real=True)

# Scalar fields
phi = sp.Function('phi')(x, y, z)
psi = sp.Function('psi')(x, y, z)

# Presence / "g"-direction vector (symbolic)
v_gx, v_gy, v_gz = sp.symbols('v_gx v_gy v_gz', real=True)
v_g = sp.Matrix([v_gx, v_gy, v_gz])

# ---------- Differential geometry bits ----------
# Gradient and Hessian of phi
grad_phi = sp.Matrix([sp.diff(phi, x), sp.diff(phi, y), sp.diff(phi, z)])
H_phi = sp.hessian(phi, (x, y, z))             # 3x3 matrix
tr_H_phi = sp.trace(H_phi)                      # trace(Hessian)

# Unit vector along grad(phi) where defined
grad_phi_norm = sp.sqrt((grad_phi.T*grad_phi)[0])
u_phi = sp.simplify(grad_phi/grad_phi_norm)     # formal unit direction (symbolic)

# G-field (choose the variant you want to test)
# Option A: "current" by removing the scalar trace along the grad direction
# G = ∇φ - (tr Hess φ) * û_phi
G_A = sp.simplify(grad_phi - tr_H_phi * u_phi)

# Option B: projection removing the component along û_phi (i.e. tangential part)
P_tan = sp.eye(3) - (u_phi*u_phi.T)             # projector orthogonal to û_phi
G_B = sp.simplify(P_tan * grad_phi)             # tangential component of ∇φ

# ---------- VOID operator as an orthogonal projector ----------
# Model VOID as a symmetric idempotent projector P (P^2 = P, P^T = P)
# Keep symbolic entries but enforce symmetry; you can set numeric ones later.
p11, p12, p13, p22, p23, p33 = sp.symbols('p11 p12 p13 p22 p23 p33', real=True)
P = sp.Matrix([[p11, p12, p13],
               [p12, p22, p23],
               [p13, p23, p33]])  # symmetric by construction

# Lyapunov integrand: 1/2 * (VOID G)·G  (choose G_A or G_B)
G = G_A  # or G_B
Lyap_density = sp.simplify(sp.Rational(1,2) * (G.T * P * G)[0])

# ---------- Phase-lock: directional derivative along v_g ----------
# ∇_g ψ = v_g · ∇ψ = 0
grad_psi = sp.Matrix([sp.diff(psi, x), sp.diff(psi, y), sp.diff(psi, z)])
phase_lock_eq = sp.Eq((v_g.T * grad_psi)[0], 0)

# ---------- VOID constraint: ∫ VOID I_g dβ = 0 (symbolic) ----------
I_g = sp.Function('I_g')(beta)  # some current along beta
VOID_constraint = sp.Eq(sp.Integral(P * I_g, (beta, sp.Symbol('beta0'), sp.Symbol('beta1'))), sp.zeros(3,1))

# ---------- Limit check: (1 + 1/n)^n -> e ----------
n = sp.symbols('n', positive=True)
limit_expr = sp.limit((1 + 1/n)**n, n, sp.oo)  # should be E
# print(limit_expr)  # uncomment to see 'E'

# ---------- Riemann xi(s) and S(σ,t) ----------
sigma, t = sp.symbols('sigma t', real=True)
s = sigma + sp.I*t
# xi(s) = 1/2 s(s-1) π^{-s/2} Γ(s/2) ζ(s)
xi = sp.Rational(1,2) * s*(s-1) * sp.pi**(-s/2) * sp.gamma(s/2) * sp.zeta(s)

# S(σ,t) = ∂_σ log |xi(σ+it)|  (symbolic form; derivative of real log-abs)
S = sp.diff(sp.log(sp.Abs(xi)), sigma)

# Optional: normal-equation surrogate for monotonicity check (EVT "MMC"-style)
dS_dsigma = sp.diff(S, sigma)

# ---------- Pretty output helpers ----------
to_show = {
    "grad_phi": grad_phi,
    "Hessian_phi": H_phi,
    "trace(H_phi)": tr_H_phi,
    "G_A": G_A,
    "G_B": G_B,
    "Lyapunov_density": sp.simplify(Lyap_density),
    "Phase_lock (v_g · ∇psi = 0)": phase_lock_eq,
    "VOID_constraint": VOID_constraint,
    "limit (1+1/n)^n": limit_expr,
    "xi(s)": sp.simplify(xi),
    "S(sigma,t) = ∂σ log|xi|": sp.simplify(S),
    "∂σ S": sp.simplify(dS_dsigma),
}
for k,v in to_show.items():
    print(f"\n--- {k} ---\n{sp.pretty(v)}")
