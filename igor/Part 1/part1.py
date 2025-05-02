import numpy as np
from numpy.polynomial.hermite import hermval
from numpy import trapz
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.special import factorial
from qdyn import animate_dynamics, propagator
import math


# Empirical constant
b = 1.065
a0 = 1
N_a = 6.02214e23
m_e = 1
h_bar = 1

#masses in g/mol
u_to_au = 1822.888486
I127_mass_au = 126.90447 * u_to_au
Cl35_mass_au = 34.968852 * u_to_au
Cl37_mass_au = 36.965903 * u_to_au

red_mass1 = (I127_mass_au * Cl35_mass_au) / (I127_mass_au + Cl35_mass_au)
red_mass2 = (I127_mass_au * Cl37_mass_au) / (I127_mass_au + Cl37_mass_au)

# Parameters for each electronic state

#X1Σ+ state
D_e_0= 0.079999   # in E_h
r_e_0= 4.3858    # in a_0
gamma_0= 0.9790  # in a_0^-1
T_e_0= 0.10       # in E_h

#A3Π1 state
D_e_1= 0.017385   # in E_h
r_e_1= 5.0877    # in a_0
gamma_1= 1.1602 # in a_0^-1
T_e_1= 0.1626   # in E_h

#propagation parameters
dt = 2
nsteps = 2000
x_grid = np.linspace(2.5, 10.5, 1000)



def V(x, D_e, gamma, r_e, b, T_e):
    x = np.array(x, dtype=np.float64)  # Allow for vectorized input
    term1 = 1 - np.exp(-gamma**2 * r_e / x * (x - r_e)**2)
    term2 = 1 - (4/5) * (b * gamma - b**2 / r_e) * np.sqrt(r_e / x) * (x - r_e) * np.exp(-b * gamma * (x - r_e))
    return D_e * term1 * term2 + T_e


def dipole(x):
    return -0.036 *  x**3 + 0.428 * x**2 -1.64 * x + 2.03


def eigen_ho(x, v, m, k):
    """Calculates the eigenfunction of the harmonic oscillator system.

    Arguments
    x: is a space coordinate.
    v: is the vibrational quantum number.
    m: is the mas of the system.
    k: is the force constant of the harmonic potential.
    """

    hermite_sum = np.zeros(v + 1)
    hermite_sum[-1] = 1
    return 1 / (2 ** v * math.factorial(v)) ** 0.5 * (((m * k) ** 0.5) / np.pi) ** 0.25 * np.e ** (
                -x ** 2 * ((m * k) ** 0.5) / 2) * hermval((m * k) ** 0.25 * x, hermite_sum)



ground_state_force_cst = 2 * D_e_0 * gamma_0**2
psi0 = eigen_ho(x_grid - r_e_0, 0, red_mass1, ground_state_force_cst)
psi0_vis = 0.01 * psi0 + V(r_e_0, D_e_0, gamma_0, r_e_0, b, T_e_0)

V0 = V(x_grid, D_e_0, gamma_0, r_e_0, b, T_e_0)
V1 = V(x_grid, D_e_1, gamma_1, r_e_1, b, T_e_1)

dipole_moment = dipole(x_grid - r_e_0)
dipole_moment_vis = 0.02 * dipole_moment + V(r_e_0, D_e_0, gamma_0, r_e_0, b, T_e_0)

#propagation
wf_dynamics=np.zeros((nsteps+1,len(x_grid)))
wf_dynamics[0]=psi0
for step in range(nsteps):
    psi = propagator(
        x_grid, wf_dynamics[step], red_mass1, dt,
        lambda x: V(x, D_e_1, gamma_1, r_e_1, b, T_e_1))
    wf_dynamics[step+1]=psi


autocorr = np.array([
    trapz(np.conj(wf_dynamics[0]) * wf_dynamics[i], x_grid)
    for i in range(wf_dynamics.shape[0])
])

# Time axis
time_array = np.linspace(0, nsteps * dt, nsteps + 1)

# Plot real, imaginary, and magnitude
plt.figure(figsize=(8, 5))
plt.plot(time_array, np.real(autocorr), label='Re[Autocorr]', color='blue')
plt.plot(time_array, np.imag(autocorr), label='Im[Autocorr]', color='red')
plt.plot(time_array, np.abs(autocorr), label='|Autocorr|', color='black', linestyle='--')
plt.xlabel("Time (a.u.)")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Function vs. Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




figure = plt.figure(figsize=(8, 5))
plt.plot(x_grid, V(x_grid,0.079999, 0.9790, 4.3858, 1.065, 0.10), label='Ground')
plt.plot(x_grid, V(x_grid, 0.017385, 1.1602, 5.0877, 1.065, 0.1626), label='Excited')
plt.plot(x_grid, psi0_vis, label='Ground State Wavefunction (scaled)')
plt.plot(x_grid, dipole_moment_vis, label='Dipole moment')


plt.xlabel('$x/a_0$')
plt.ylabel('$V/E_h$')
plt.title('Potential Energy Curves and Ground State Wavefunction')
plt.ylim(0, 0.3)

plt.legend()
plt.grid()
plt.show()

# after propagation loop, but before animate:
for i in range(wf_dynamics.shape[0]):
    # 1) shift the packet so it’s plotted around r_e_1, not x=0
    wf = wf_dynamics[i]
    shifted = wf  # psi_n was built with x_grid - r_e_0, but now on V1 it's fine

    # 2) normalize (just to be safe)
    shifted /= np.sqrt(np.sum(np.abs(shifted) ** 2) * (x_grid[1] - x_grid[0]))

    # 3) scale into the energy plot
    baseline = V(r_e_1, D_e_1, gamma_1, r_e_1, b, T_e_1)
    wf_dynamics[i] = 0.02 * shifted + baseline

ani = animate_dynamics(
    x_grid,
    wf_dynamics,
    dt,
    V(x_grid, D_e_1, gamma_1, r_e_1, b, T_e_1)
)

plt.show()

plt.figure(figsize=(8, 5))
plt.plot(x_grid, autocorr)
plt.show()