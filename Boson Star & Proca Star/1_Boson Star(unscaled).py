import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Global variable for initial scalar field value
phi_c_values = [0.07, 0.04, 0.02, 0.01]

# Planck mass squared 
M_pl_squared = 1.0
m = 1
G = 1

# Function defining the differential equations
def bsode(r, y, p):
    omega = p[0]

    a = y[0]
    alpha = y[1]
    psi0 = y[2]
    phi = y[3]
    
    # ds^2 = -alpha(r)^2 dt^2 +a(r)^2 dr^2 + r d Omega^2
    
    dydr = np.zeros_like(y)
    dydr[0] = (1/2) * ((a/r) * ((1 - a**2) + 4 * np.pi * G * r * a * (psi0**2 * a**2 * (m**2 + omega**2 / alpha**2) + phi**2)))
    dydr[1] = (alpha/2) * (((a**2 - 1)/r + 4 * np.pi * r * (psi0**2 * a**2 * (omega**2 / alpha**2 - m**2) + phi**2)))
    dydr[2] = phi
    dydr[3] = -(1 + a**2 - 4 * np.pi * r**2 * psi0**2 * a**2 * m**2) * (phi/r) - (omega**2 / alpha**2 - m**2) * psi0 * a**2

    return dydr

# Function defining the boundary conditions
def bsbc(ya, yb, p):
    res = np.array([ya[0] - 1,  # a(0) = 1
                    ya[1] - 1,  # alpha(0) = 1
                    ya[2] - phi_c,  # psi0(0) = phi_c
                    yb[2],  # psi0(infinity) = 0
                    ya[3]])  # phi(0) = 0
    return res

# Function scaling alpha and omega
def scale_lapse_and_frequency(f, i, sol_parameters):
    alpha_last = 1 / f[1, i]  # scaling factor for alpha at large r
    scaled_alpha = alpha_last * f[1, :i]  # Apply scaling to alpha(r)
    scaled_omega = alpha_last * sol_parameters[0]  # Adjusting omega to maintain physical relationship
    return scaled_alpha, scaled_omega

def ADM_mass(r, a):
    M = (1 - a**-2) * (r / 2)
    return M

# Initial setup and loop over different phi_c
omega = 0.8
x_init = np.linspace(1e-5, 15, 1000)  # radial grid

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for phi_c in phi_c_values:
    solinit = np.vstack((np.ones_like(x_init), np.ones_like(x_init),
                         phi_c * np.ones_like(x_init), np.zeros_like(x_init)))  # initial guesses
    sol = solve_bvp(bsode, bsbc, x_init, solinit, p=[omega], tol=1e-6)

    f = sol.y  # solution
    i = np.argmax(f[2, :] < 1e-5)

    r = sol.x[:i]
    a = f[0, :i]  # `a` is now used consistently instead of `a1`
    scaled_alpha, scaled_omega = scale_lapse_and_frequency(f, i, sol.p)
    M = ADM_mass(r, a)

    # Scaling M by Mmax
    M_max = 0.633 * M_pl_squared / m
    M_scaled = M / M_max

    # Plotting results
    axs[0].plot(r, scaled_alpha, label=f'phi_c = {phi_c}')
    axs[1].plot(r, f[2, :i], label=f'phi_c = {phi_c}')
    axs[2].plot(r, M_scaled, label=f'phi_c = {phi_c}')

# Titles and labels
axs[0].set_title('Lapse Function (alpha) vs Radial Coordinate (r)')
axs[0].set_xlabel('r')
axs[0].set_ylabel('alpha(r)')

axs[1].set_title('Scalar Field (psi0) vs Radial Coordinate (r)')
axs[1].set_xlabel('r')
axs[1].set_ylabel('psi0(r)')

axs[2].set_title('Scaled ADM Mass (M) vs Radial Coordinate (r)')
axs[2].set_xlabel('r')
axs[2].set_ylabel('M')

plt.legend()
plt.tight_layout()
plt.show()
