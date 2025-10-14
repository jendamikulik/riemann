import mpmath as mp

mp.dps = 50  # nastav přesnost

def xi_eval(s):
    # xi(s) = 1/2 s(s-1) pi^{-s/2} Gamma(s/2) zeta(s)
    return 0.5*s*(s-1) * mp.pi**(-s/2) * mp.gamma(s/2) * mp.zeta(s)

def S_sigma(sigma, t):
    # S(σ,t) = d/dσ log |xi(σ + i t)|  (komplexní diferenciální krok)
    s0 = sigma + 1j*t
    h  = mp.mpf('1e-8')
    xi_p = xi_eval(s0 + h)
    xi_m = xi_eval(s0 - h)
    # derivace log|xi| ~ Re( (xi'/xi) ) * (dσ), ale numericky uděláme centrální krok:
    logabs_p = mp.log(abs(xi_p))
    logabs_m = mp.log(abs(xi_m))
    return (logabs_p - logabs_m) / (2*h)

def dS_dsigma(sigma, t):
    h = mp.mpf('1e-6')
    return (S_sigma(sigma+h, t) - S_sigma(sigma-h, t)) / (2*h)

def find_root_near_half(t):
    f = lambda sig: S_sigma(sig, t)
    try:
        # hledej kořen poblíž 0.5 (okno ±0.1)
        return mp.findroot(f, (0.45, 0.55))
    except:  # fallback na hrubší scan
        sigs = [0.45 + k*(0.10/50) for k in range(51)]
        vals = [f(s) for s in sigs]
        for a,b,fa,fb in zip(sigs, sigs[1:], vals, vals[1:]):
            if fa == 0 or fa*fb < 0:
                return mp.findroot(f, (a,b))
        return None

# příklad: první Riemannova nula ~ t1 = 14.134725...
t_val = mp.mpf('14.134725141734693790')
sig_star = find_root_near_half(t_val)
print("sigma*(t) ~", sig_star)
if sig_star is not None:
    print("S(sigma*, t) ~", S_sigma(sig_star, t_val))
    print("dS/dsigma(sigma*, t) ~", dS_dsigma(sig_star, t_val))
