import numpy as np
from numpy.polynomial.hermite import hermval
from numpy import trapz
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from qdyn import animate_dynamics, propagator
import math

# Constantes
b = 1.065
u_to_au = 1822.888486
I127_mass_au = 126.90447 * u_to_au
Cl35_mass_au = 34.968852 * u_to_au
Cl37_mass_au = 36.965903 * u_to_au

# Masse réduite
red_mass1 = (I127_mass_au * Cl35_mass_au) / (I127_mass_au + Cl35_mass_au)

# Paramètres des potentiels
D_e_0, r_e_0, gamma_0, T_e_0 = 0.079999, 4.3858, 0.9790, 0.10
D_e_1, r_e_1, gamma_1, T_e_1 = 0.017385, 5.0877, 1.1602, 0.1626

# Grilles
dt = 1
nsteps = 7000
x_grid = np.linspace(2.5, 15, 1500)
dx = x_grid[1] - x_grid[0]

# Fonctions utiles
def V(x, D_e, gamma, r_e, b, T_e):
    x = np.array(x, dtype=np.float64)
    term1 = 1 - np.exp(-gamma**2 * r_e / x * (x - r_e)**2)
    term2 = 1 - (4/5) * (b * gamma - b**2 / r_e) * np.sqrt(r_e / x) * (x - r_e) * np.exp(-b * gamma * (x - r_e))
    return D_e * term1 * term2 + T_e

def dipole(x):
    return -0.036 * x**3 + 0.428 * x**2 - 1.64 * x + 2.03

def eigen_ho(x, v, m, k):
    hermite_sum = np.zeros(v + 1)
    hermite_sum[-1] = 1
    return 1 / (2 ** v * math.factorial(v))**0.5 * ((m * k)**0.5 / np.pi)**0.25 * np.exp(-x**2 * ((m * k)**0.5) / 2) * hermval((m * k)**0.25 * x, hermite_sum)

# Fonction d'onde initiale
k = 2 * D_e_0 * gamma_0**2
psi0 = eigen_ho(x_grid - r_e_0, 0, red_mass1, k)
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx) 

# Potentiels
V0 = V(x_grid, D_e_0, gamma_0, r_e_0, b, T_e_0)
V1 = V(x_grid, D_e_1, gamma_1, r_e_1, b, T_e_1)

# Dipôle et projection
mu = dipole(x_grid - r_e_0)
phi0 = mu * psi0

# Propagation
wf_dynamics = np.zeros((nsteps + 1, len(x_grid)), dtype=np.complex128)
wf_dynamics[0] = phi0
for step in range(nsteps):
    wf_dynamics[step + 1] = propagator(x_grid, wf_dynamics[step], red_mass1, dt, lambda x: V(x, D_e_1, gamma_1, r_e_1, b, T_e_1))

# Autocorrélation
autocorr = np.array([
    trapz(np.conj(wf_dynamics[0]) * wf_dynamics[i], x_grid)
    for i in range(nsteps + 1)
])
time_array = np.linspace(0, nsteps * dt, nsteps + 1)

# === PLOTS ===

# Potentiel + ψ0 + µ
baseline = V(r_e_0, D_e_0, gamma_0, r_e_0, b, T_e_0)
psi0_vis = 0.01 * psi0 + baseline
mu_vis = 0.02 * mu + baseline

plt.figure(figsize=(8, 5))
plt.plot(x_grid, V0, label='Ground')
plt.plot(x_grid, V1, label='Excited')
plt.plot(x_grid, psi0_vis, label='ψ₀ (scaled)')
plt.plot(x_grid, mu_vis, label='µ (scaled)')
plt.xlabel('$x/a_0$')
plt.ylabel('$V/E_h$')
plt.title('Potentiels, ψ₀ et µ')
plt.ylim(0, 0.3)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Autocorrélation
plt.figure(figsize=(8, 5))
plt.plot(time_array, np.real(autocorr), label='Re[C(t)]')
plt.plot(time_array, np.imag(autocorr), label='Im[C(t)]')
plt.plot(time_array, np.abs(autocorr), label='|C(t)|', linestyle='--')
plt.xlabel("Temps (a.u.)")
plt.ylabel("Autocorrélation")
plt.title("Fonction d’autocorrélation")
plt.ylim(-5, 5) # for |C(t)|
plt.xlim(0, 750)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Spectre
spectrum = np.fft.fftshift(np.fft.fft(autocorr))
freqs = np.fft.fftshift(np.fft.fftfreq(len(time_array), d=dt)) * 2 * np.pi

plt.figure(figsize=(8, 5))
plt.plot(freqs, np.abs(spectrum)**2)
plt.xlabel("Fréquence (a.u.)")
plt.ylabel("Intensité")
plt.title("Spectre d’absorption par FFT")
plt.grid(True)
plt.tight_layout()
plt.show()

# Animation
for i in range(wf_dynamics.shape[0]):
    wf = wf_dynamics[i]
    wf /= np.sqrt(np.sum(np.abs(wf)**2) * dx)
    baseline = V(r_e_1, D_e_1, gamma_1, r_e_1, b, T_e_1)
    wf_dynamics[i] = np.real(0.02 * wf + baseline)

ani = animate_dynamics(x_grid, wf_dynamics, dt, V1)
plt.show()