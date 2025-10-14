# Clean, runnable demo of the sign-flux profile S(σ,t) without external deps.
# (No seaborn, one chart, no explicit colors.)

import numpy as np
import matplotlib.pyplot as plt

# Parameters
t = 10.0  # fixed imaginary part
sigma = np.linspace(0.0, 1.0, 1200)

# Asymptotic main term and a small, smooth "error" term
A_t = 0.5 * np.log(t / (2 * np.pi))         # ~ (1/2) log(|t|/2π)
E = 0.1 * np.sin(2 * np.pi * sigma)         # toy error, bounded and smooth

# Sign-flux profile
S = (sigma - 0.5) * A_t + E

# Plot
plt.figure(figsize=(9, 4))
plt.plot(sigma, S, label=f"S(σ, t={t:g})")
plt.axhline(0, linewidth=1)
plt.axvline(0.5, linestyle="--", linewidth=1, label="σ = 1/2")
plt.xlabel("σ (real part)")
plt.ylabel("S(σ, t)")
plt.title("Sign-flux function S(σ, t) = A(t)(σ − 1/2) + E(σ,t)")
plt.legend()
plt.tight_layout()
plt.show()

# Save a copy for inclusion in papers
#out_path = "/mnt/data/sign_flux_demo.png"
#plt.savefig(out_path, dpi=200)
#out_path
