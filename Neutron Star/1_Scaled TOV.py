import numpy as np
import matplotlib.pyplot as plt

def fdmdx(x, m, p, G, c, M1, rho1, P1, X1):
    rho = frho(x, m, p, G, c, M1, rho1, P1, X1)
    dmdx = (4 * np.pi * rho1 * X1**3 / M1) * rho * x**2
    return dmdx

def fdpdx(x, m, p, G, c, M1, rho1, P1, X1):
    rho = frho(x, m, p, G, c, M1, rho1, P1, X1)
    dpdx = -(G * M1 * rho1 / (X1 * P1)) * (m * rho / x**2) * (1 + (P1 / (rho1 * c**2)) * (p / rho)) * \
           (1 + (4 * np.pi * X1**3 * P1 / (M1 * c**2)) * (x**3 * p / m)) * (1 - (2 * G * M1 / (X1 * c**2)) * (m / x))**(-1)
    return dpdx

def frho(x, m, p, G, c, M1, rho1, P1, X1):
    gam = 1.8
    return (p**(1 / gam)) + p / (gam - 1)

def main():
    # Initializing parameters and constants
    x = 1e-5
    dx = 1e-4
    m = 1e-40
    p = 1.0
    pdr = 0

    G = 6.674e-8  # cgs units
    c = 3e10  # cgs units
    M1 = 2*1e33
    rho1 = 1e15
    P1 = 1e35
    X1 = 1e5

    xvec = []
    mvec = []
    pvec = []
    dpvec = []

    # Calculating mass, pressure, and scalar field using the 4th-order Runge-Kutta method
    while p > 1e-50:
        km1 = dx * fdmdx(x, m, p, G, c, M1, rho1, P1, X1)
        kp1 = dx * fdpdx(x, m, p, G, c, M1, rho1, P1, X1)
        #k1 = fdpdx(x, m, p, G, c, M1, rho1, P1, X1)

        if (p + kp1 / 2) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km2 = dx * fdmdx(x + dx / 2, m + km1 / 2, p + kp1 / 2, G, c, M1, rho1, P1, X1)
        kp2 = dx * fdpdx(x + dx / 2, m + km1 / 2, p + kp1 / 2, G, c, M1, rho1, P1, X1)
        #k2 = fdpdx(x + dx / 2, m + km1 / 2, p + kp1 / 2, G, c, M1, rho1, P1, X1)

        if (p + kp2 / 2) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km3 = dx * fdmdx(x + dx / 2, m + km2 / 2, p + kp2 / 2, G, c, M1, rho1, P1, X1)
        kp3 = dx * fdpdx(x + dx / 2, m + km2 / 2, p + kp2 / 2, G, c, M1, rho1, P1, X1)
        #k3 = fdpdx(x + dx / 2, m + km2 / 2, p + kp2 / 2, G, c, M1, rho1, P1, X1)

        if (p + kp3) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km4 = dx * fdmdx(x + dx, m + km3, p + kp3, G, c, M1, rho1, P1, X1)
        kp4 = dx * fdpdx(x + dx, m + km3, p + kp3, G, c, M1, rho1, P1, X1)
        #k4 = fdpdx(x + dx, m + km3, p + kp3, G, c, M1, rho1, P1, X1)

        dm = (km1 / 6 + km2 / 3 + km3 / 3 + km4 / 6)
        m += dm

        dp = (kp1 / 6 + kp2 / 3 + kp3 / 3 + kp4 / 6)
        p += dp

        #dpdr = (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        #pdr += dpdr

        if p < 0:
            x -= dx
            m -= dm
            p -= dp
            #pdr -= dpdr
            dx /= 10

        xvec.append(x)
        mvec.append(m)
        pvec.append(p)
        dpvec.append(pdr)

        x += dx

    #PLOT PRESSURE VS RADIUS
    plt.subplot(1,2,1)
    plt.plot(xvec,pvec)
    plt.xlabel("Radius(km)")
    plt.ylabel("Dimensionless Pressure")
    plt.title("P-r plot")

    #PLOT MASS VS RADIUS
    plt.subplot(1,2,2)
    plt.plot(xvec,mvec)
    plt.xlabel("Radius(km)")
    plt.ylabel("Dimensionless Mass")
    plt.title("M-r plot")
    
    plt.show()

if __name__ == "__main__":
    main() 