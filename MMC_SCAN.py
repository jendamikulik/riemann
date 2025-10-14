
from mpmath import mp, zeta, gamma, pi
import numpy as np
mp.dps = 70
def xi(s):
    return mp.mpf('0.5')*s*(s-1)*mp.power(pi, -s/2)*gamma(s/2)*zeta(s)
def S_sigma(sigma, t, h=1e-6):
    s1 = (sigma + h) + 1j*t
    s2 = (sigma - h) + 1j*t
    xi1 = xi(s1)
    xi2 = xi(s2)
    return (mp.log(abs(xi1)) - mp.log(abs(xi2))) / (2*h)
def S_prime_sigma(sigma, t, h=1e-4):
    return (S_sigma(sigma+h, t) - S_sigma(sigma-h, t)) / (2*h)
def mmc_scan(t, sigma_min=0.05, sigma_max=0.95, grid=400, tol=1e-10, maxiter=80):
    sigmas = np.linspace(sigma_min, sigma_max, grid)
    Svals = [float(S_sigma(float(s), t)) for s in sigmas]
    sign_idx = None
    for i in range(len(sigmas)-1):
        if Svals[i] == 0.0 or Svals[i]*Svals[i+1] < 0:
            sign_idx = i
            break
    if sign_idx is None:
        return {"status":"no-crossing"}
    a = float(sigmas[sign_idx]); b = float(sigmas[sign_idx+1])
    Sa = float(S_sigma(a, t)); Sb = float(S_sigma(b, t))
    it=0
    while (b-a)>tol and it<maxiter:
        m = 0.5*(a+b)
        Sm = float(S_sigma(m, t))
        if Sa*Sm <= 0: b, Sb = m, Sm
        else: a, Sa = m, Sm
        it += 1
    sigma_star = 0.5*(a+b)
    slope = float(S_prime_sigma(sigma_star, t))
    return {"status":"ok","sigma_star":sigma_star,"slope":slope,"iterations":it}
res = mmc_scan(20.0)
print(res)  # {'status':'ok','sigma_star': ..., 'slope': ..., 'iterations': ...}