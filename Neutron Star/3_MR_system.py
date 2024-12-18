import numpy as np
import matplotlib.pyplot as plt

def fdmdx(x, m, p, G, c, M1, rho1, P1, X1):
    rho = frho(p)
    dmdx = (4 * np.pi * rho1 * X1**3 / M1) * rho * x**2
    return dmdx

def fdpdx(x, m, p, G, c, M1, rho1, P1, X1):
    rho = frho(p)
    dpdx = -(G * M1 * rho1 / (X1 * P1)) * (m * rho / x**2) * (1 + (P1 / (rho1 * c**2)) * (p / rho)) * \
           (1 + (4 * np.pi * X1**3 * P1 / (M1 * c**2)) * (x**3 * p / m)) * (1 - (2 * G * M1 / (X1 * c**2)) * (m / x))**(-1)
    return dpdx

def frho(p):
    gam = 1.8
    rho = (p**(1/gam))  + p / (gam - 1)
    return rho

def main():
    # Initializing parameters and constants
    G = 6.674e-8  # cgs units
    c = 3e10  # cgs units
    M1 = 1e33
    rho1 = 1e15
    P1 = 1e35
    X1 = 1e6

    initial_pressures = np.logspace(np.log10(1e-10),np.log10(10), 1000)  # Range of initial pressures
    #final_masses = []
    #final_radii = []

    pressures = [] 
    masses = []
    radii = []

    for initial_p in initial_pressures:
        x = 1e-5
        dx = 1e-3
        m = 1e-40
        p = initial_p
        pdr = 0

        while p > 1e-50:
            km1 = dx * fdmdx(x, m, p, G, c, M1, rho1, P1, X1)
            kp1 = dx * fdpdx(x, m, p, G, c, M1, rho1, P1, X1)
            k1 = fdpdx(x, m, p, G, c, M1, rho1, P1, X1)

            if (p + kp1 / 2) < 0:
                x -= dx
                dx /= 10
                x += dx
                continue

            km2 = dx * fdmdx(x + dx / 2, m + km1 / 2, p + kp1 / 2, G, c, M1, rho1, P1, X1)
            kp2 = dx * fdpdx(x + dx / 2, m + km1 / 2, p + kp1 / 2, G, c, M1, rho1, P1, X1)
            k2 = fdpdx(x + dx / 2, m + km1 / 2, p + kp1 / 2, G, c, M1, rho1, P1, X1)

            if (p + kp2 / 2) < 0:
                x -= dx
                dx /= 10
                x += dx
                continue

            km3 = dx * fdmdx(x + dx / 2, m + km2 / 2, p + kp2 / 2, G, c, M1, rho1, P1, X1)
            kp3 = dx * fdpdx(x + dx / 2, m + km2 / 2, p + kp2 / 2, G, c, M1, rho1, P1, X1)
            k3 = fdpdx(x + dx / 2, m + km2 / 2, p + kp2 / 2, G, c, M1, rho1, P1, X1)

            if (p + kp3) < 0:
                x -= dx
                dx /= 10
                x += dx
                continue

            km4 = dx * fdmdx(x + dx, m + km3, p + kp3, G, c, M1, rho1, P1, X1)
            kp4 = dx * fdpdx(x + dx, m + km3, p + kp3, G, c, M1, rho1, P1, X1)
            k4 = fdpdx(x + dx, m + km3, p + kp3, G, c, M1, rho1, P1, X1)

            dm = (km1 / 6 + km2 / 3 + km3 / 3 + km4 / 6)
            m += dm

            dp = (kp1 / 6 + kp2 / 3 + kp3 / 3 + kp4 / 6)
            p += dp

            dpdr = (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
            pdr += dpdr

            if p < 0:
                x -= dx
                m -= dm
                p -= dp
                pdr -= dpdr
                dx /= 10

            x += dx


        pressures.append(initial_p)
        masses.append(m)
        radii.append(x)

    masses = np.array(masses)
    radii = np.array(radii)

        # Output the mass and radius values
    print(f"Initial pressure: {initial_p:.2e}, Final mass: {m:.2e}, Final radius: {x:.2e}")

    # Plotting the mass-radius relation
    plt.figure()
    plt.plot(radii, masses)
    plt.title('Mass-Radius profile for a neutron star')
    plt.xlabel('Dimensionless Radius')
    plt.ylabel('Dimensionless Mass')
    plt.grid()
    plt.show()

    file_path = r'C:\Users\titir\OneDrive\Desktop\IUCAA PROJECT\MR & k2_C PLOT\mass_radius_data_x.txt'
    with open(file_path, 'w') as f:
        for p, m, r in zip(pressures, masses, radii):
            f.write(f"{p:.2e}\t{m:.2e}\t{r:.2e}\n")

if __name__ == "__main__":
    main()