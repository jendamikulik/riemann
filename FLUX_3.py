import numpy as np
import matplotlib.pyplot as plt
from mpmath import zeta, zeta

# Nastavení rozsahu
x = np.linspace(-25, 25, 2000)  # Imaginární část
s = [0.5 + 1j * xi for xi in x]  # Body na kritické přímce

# Spočítej hodnoty ζ(1/2 + ix)
zeta_vals = [complex(zeta(si)) for si in s]

# Reálná a imaginární část
zeta_real = [z.real for z in zeta_vals]
zeta_imag = [z.imag for z in zeta_vals]

# --- Vizualizace ---
plt.figure(figsize=(12, 5))
plt.plot(x, zeta_real, color="#e69f00", label=r'$\Re \zeta(1/2 + ix)$', linewidth=2)
plt.plot(x, zeta_imag, color="#56b4e9", label=r'$\Im \zeta(1/2 + ix)$', linewidth=2)

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

plt.xlabel(r'$x$', fontsize=12)
plt.ylabel('value', fontsize=12)
plt.title(r'Real and Imag parts of $\zeta(1/2 + ix)$ on the critical line', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

# --- Ulož výsledek ---
plt.savefig("riemann_zeta_critical_line.png", dpi=300)
plt.show()
