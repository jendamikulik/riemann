import sympy as sp
x,y,z, kappa = sp.symbols('x y z kappa', real=True)
phi = sp.Rational(1,2)*(x**2 + y**2) + kappa*z

grad_phi = sp.Matrix([sp.diff(phi,x), sp.diff(phi,y), sp.diff(phi,z)])
H_phi    = sp.hessian(phi, (x,y,z))
tr_H     = sp.trace(H_phi)
grad_norm= sp.sqrt((grad_phi.T*grad_phi)[0])
u = sp.simplify(grad_phi/grad_norm)  # směr ∇φ

# G = ∇φ - tr(Hφ) * u
G = sp.simplify(grad_phi - tr_H*u)

P = sp.diag(1,1,0)  # VOID projektor
Lyap_density = sp.simplify(sp.Rational(1,2)*(G.T*P*G)[0])
print("Lyapunov density:", sp.simplify(Lyap_density))
