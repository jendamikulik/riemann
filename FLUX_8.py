import mpmath as mp

mp.dps = 80  # zvýšená přesnost (klidně 60–120 dle potřeby)

# --- pomocné spec. funkce ---
psi  = mp.digamma      # digamma
psi1 = mp.polygamma    # polygamma(n, z), zde použijeme n=1 pro trigammu

def L(s):
    """
    L(s) = xi'(s)/xi(s) = 1/s + 1/(s-1) - (1/2)ln(pi) + (1/2)psi(s/2) + zeta'(s)/zeta(s)
    """
    term_basic = 1/s + 1/(s-1) - 0.5*mp.log(mp.pi) + 0.5*psi(s/2)
    # log-derivative ζ'(s)/ζ(s) – robustně přes centrální rozdíl s adaptivním krokem
    h = mp.mpf('1e-6')  # rozumný krok v reálném směru
    zph = mp.zeta(s+h)
    zmh = mp.zeta(s-h)
    z   = mp.zeta(s)
    zprime_over_z = ((zph - zmh)/(2*h)) / z
    return term_basic + zprime_over_z

def S_sigma(sigma, t):
    s = sigma + 1j*t
    return mp.re(L(s))

def dS_dsigma(sigma, t):
    # derivuj už L(s) (hladší než log|xi|); jemný centrální krok
    s = sigma + 1j*t
    h = mp.mpf('1e-6')
    return mp.re((L(s+h) - L(s-h)) / (2*h))

def find_sigma_star(t, left=0.3, right=0.7, grid=400):
    # 1) nabracketuj změnu znaménka S(σ,t)
    sigs = [left + k*(right-left)/grid for k in range(grid+1)]
    vals = [S_sigma(sig, t) for sig in sigs]
    br = None
    for a,b,fa,fb in zip(sigs, sigs[1:], vals, vals[1:]):
        if fa == 0:
            return a
        if fa*fb < 0:
            br = (a,b); break
    if br is None:
        return None  # nenašlo se – rozšiř interval nebo změň t o malé ε
    # 2) Brent bez derivací
    return mp.findroot(lambda s: S_sigma(s, t), br)

# --- příklad: posuň t o malé ε od první nuly zety ---
t_ref = mp.mpf('14.134725141734693790') + mp.mpf('1e-3')
sig = find_sigma_star(t_ref)
print("sigma*(t) =", sig)
if sig is not None:
    print("S(sigma*, t) =", S_sigma(sig, t_ref))
    print("dS/dsigma(sigma*, t) =", dS_dsigma(sig, t_ref))
