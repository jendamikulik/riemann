import mpmath as mp

mp.dps = 80  # vysoká přesnost

def xi(s):
    return 0.5*s*(s-1) * mp.pi**(-s/2) * mp.gamma(s/2) * mp.zeta(s)

def xi_prime_complex_step(s, h=1e-12):
    # complex-step derivative: f'(x) ≈ Im(f(x + i h))/h
    return (mp.im(xi(s + 1j*h))) / h

def S_sigma(sigma, t):
    s = sigma + 1j*t
    num = xi_prime_complex_step(s)
    den = xi(s)
    return mp.re(num/den)

def dS_dsigma(sigma, t, h=1e-6):
    # jemná numerická derivace už nad stabilním S
    return (S_sigma(sigma+h, t) - S_sigma(sigma-h, t)) / (2*h)

def find_sigma_star(t, left=0.3, right=0.7, grid=400):
    # 1) nabracketujeme změnu znaménka S(σ,t) v intervalu
    sigs = [left + k*(right-left)/grid for k in range(grid+1)]
    vals = [S_sigma(sig, t) for sig in sigs]
    br = None
    for a,b,fa,fb in zip(sigs, sigs[1:], vals, vals[1:]):
        if fa == 0:
            return a
        if fa*fb < 0:
            br = (a,b); break
    if br is None:
        return None
    # 2) Brent (bez derivací)
    return mp.findroot(lambda s: S_sigma(s, t), br)

# Pozor na t: nepoužívej přesné ordináty nul zety.
# Vem třeba t = 14.1347251417 + 1e-3
t_val = mp.mpf('14.1347251417') + mp.mpf('1e-3')

sig_star = find_sigma_star(t_val)
print("sigma*(t) =", sig_star)
if sig_star is not None:
    print("S(sigma*, t) =", S_sigma(sig_star, t_val))
    print("dS/dsigma(sigma*, t) =", dS_dsigma(sig_star, t_val))
