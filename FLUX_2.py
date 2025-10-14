# Proof-of-symmetry demo for zeta on the critical line.
# Shows that Re ζ(1/2+ix) is even and Im ζ(1/2+ix) is odd by overlaying the curve
# with its mirror x -> -x and reporting max deviations.
import numpy as np
import matplotlib.pyplot as plt
from mpmath import zeta
# Symmetric grid
x = np.linspace(-25, 25, 3001)
s_pos = [0.5 + 1j*xi for xi in x]
s_neg = [0.5 - 1j*xi for xi in x]  # mirror (x -> -x)
# Evaluate zeta
z_pos = np.array([complex(zeta(si)) for si in s_pos])
z_neg = np.array([complex(zeta(si)) for si in s_neg])
Re_pos, Im_pos = z_pos.real, z_pos.imag
Re_neg, Im_neg = z_neg.real, z_neg.imag
# Symmetry residuals: even/odd checks
even_residual = np.max(np.abs(Re_pos - Re_neg))      # should be ~0
odd_residual  = np.max(np.abs(Im_pos + Im_neg))      # should be ~0
# Plot original and mirrored overlays
plt.figure(figsize=(12, 5))
plt.plot(x, Re_pos, label=r'$\Re\,\zeta(1/2+ix)$')
plt.plot(x, Im_pos, label=r'$\Im\,\zeta(1/2+ix)$')
plt.plot(x, Re_neg, linestyle='--', label=r'$\Re\,\zeta(1/2-ix)$ (mirror)')
plt.plot(x, -Im_neg, linestyle='--', label=r'$-\Im\,\zeta(1/2-ix)$ (mirror)')
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel(r'$x$')
plt.ylabel('value')
plt.title(r'Critical-line symmetry: overlay with $x\mapsto -x$')
plt.legend()
plt.tight_layout()
plt.show()

#out_path = "/mnt/data/zeta_critical_line_symmetry.png"
#plt.savefig(out_path, dpi=220)
#(even_residual, odd_residual, out_path)
