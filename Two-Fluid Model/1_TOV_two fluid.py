import numpy as np
import matplotlib.pyplot as plt

# Functions for the first fluid
def fdmdx1(x, m1, p1, G, c, M1, rho1, P1, X1):
    rho = frho1(x, m1, p1, G, c, M1, rho1, P1, X1)
    dmdx = (4 * np.pi * rho1 * X1**3 / M1) * rho * x**2
    return dmdx

def fdpdx1(x, m1, p1, m2, p2, G, c, M1, rho1, P1, X1):
    rho = frho1(x, m1, p1, G, c, M1, rho1, P1, X1)
    dpdx = -(G * M1 * rho1 / (X1 * P1)) * (m1 * rho / x**2) * (1 + (P1 / (rho1 * c**2)) * (p1 / rho)) * \
           (1 + (4 * np.pi * X1**3 * P1 / (M1 * c**2)) * (x**3 * p1 / m1)) * (1 - (2 * G * M1 / (X1 * c**2)) * (m1 / x))**(-1)
    return dpdx

def frho1(x, m1, p1, G, c, M1, rho1, P1, X1):
    gam = 1.8
    return (p1**(1 / gam)) + p1 / (gam - 1)

# Functions for the second fluid
def fdmdx2(x, m2, p2, G, c, M2, rho2, P2, X2):
    rho = frho2(x, m2, p2, G, c, M2, rho2, P2, X2)
    dmdx = (4 * np.pi * rho2 * X2**3 / M2) * rho * x**2
    return dmdx

def fdpdx2(x, m1, p1, m2, p2, G, c, M2, rho2, P2, X2):
    rho = frho2(x, m2, p2, G, c, M2, rho2, P2, X2)
    dpdx = -(G * M2 * rho2 / (X2 * P2)) * (m2 * rho / x**2) * (1 + (P2 / (rho2 * c**2)) * (p2 / rho)) * \
           (1 + (4 * np.pi * X2**3 * P2 / (M2 * c**2)) * (x**3 * p2 / m2)) * (1 - (2 * G * M2 / (X2 * c**2)) * (m2 / x))**(-1)
    return dpdx

def frho2(x, m2, p2, G, c, M2, rho2, P2, X2):
    gam = 2
    return (p2**(1 / gam))  + p2 / (gam - 1)

def main():
    # Initializing parameters and constants for fluid 1
    x = 1e-5
    dx = 1e-4
    m1 = 1e-40
    p1 = 1.0
    m2 = 1e-40
    p2 = 1.0

    G = 6.674e-8  # cgs units
    c = 3e10  # cgs units

    # Constants for fluid 1
    M1 = 2 * 1e33
    rho1 = 1e15
    P1 = 1e35
    X1 = 1e5

    # Constants for fluid 2
    M2 = 2 * 1e33
    rho2 = 1e15
    P2 = 1e35
    X2 = 1e5

    xvec = []
    m1vec = []
    p1vec = []
    m2vec = []
    p2vec = []
    m_total_vec = []
    p_total_vec = []

    # Calculating mass and pressure for both fluids using the 4th-order Runge-Kutta method
    while p1 > 1e-50 and p2 > 1e-50:
        # Fluid 1 calculations
        km11 = dx * fdmdx1(x, m1, p1, G, c, M1, rho1, P1, X1)
        kp11 = dx * fdpdx1(x, m1, p1, m2, p2, G, c, M1, rho1, P1, X1)

        if (p1 + kp11 / 2) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km12 = dx * fdmdx1(x + dx / 2, m1 + km11 / 2, p1 + kp11 / 2, G, c, M1, rho1, P1, X1)
        kp12 = dx * fdpdx1(x + dx / 2, m1 + km11 / 2, p1 + kp11 / 2, m2, p2, G, c, M1, rho1, P1, X1)

        if (p1 + kp12 / 2) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km13 = dx * fdmdx1(x + dx / 2, m1 + km12 / 2, p1 + kp12 / 2, G, c, M1, rho1, P1, X1)
        kp13 = dx * fdpdx1(x + dx / 2, m1 + km12 / 2, p1 + kp12 / 2, m2, p2, G, c, M1, rho1, P1, X1)

        if (p1 + kp13) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km14 = dx * fdmdx1(x + dx, m1 + km13, p1 + kp13, G, c, M1, rho1, P1, X1)
        kp14 = dx * fdpdx1(x + dx, m1 + km13, p1 + kp13, m2, p2, G, c, M1, rho1, P1, X1)

        dm1 = (km11 / 6 + km12 / 3 + km13 / 3 + km14 / 6)
        m1 += dm1

        dp1 = (kp11 / 6 + kp12 / 3 + kp13 / 3 + kp14 / 6)
        p1 += dp1

        # Fluid 2 calculations
        km21 = dx * fdmdx2(x, m2, p2, G, c, M2, rho2, P2, X2)
        kp21 = dx * fdpdx2(x, m1, p1, m2, p2, G, c, M2, rho2, P2, X2)

        if (p2 + kp21 / 2) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km22 = dx * fdmdx2(x + dx / 2, m2 + km21 / 2, p2 + kp21 / 2, G, c, M2, rho2, P2, X2)
        kp22 = dx * fdpdx2(x + dx / 2, m1, p1, m2 + km21 / 2, p2 + kp21 / 2, G, c, M2, rho2, P2, X2)

        if (p2 + kp22 / 2) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km23 = dx * fdmdx2(x + dx / 2, m2 + km22 / 2, p2 + kp22 / 2, G, c, M2, rho2, P2, X2)
        kp23 = dx * fdpdx2(x + dx / 2, m1, p1, m2 + km22 / 2, p2 + kp22 / 2, G, c, M2, rho2, P2, X2)

        if (p2 + kp23) < 0:
            x -= dx
            dx /= 10
            x += dx
            continue

        km24 = dx * fdmdx2(x + dx, m2 + km23, p2 + kp23, G, c, M2, rho2, P2, X2)
        kp24 = dx * fdpdx2(x + dx, m1, p1, m2 + km23, p2 + kp23, G, c, M2, rho2, P2, X2)

        dm2 = (km21 / 6 + km22 / 3 + km23 / 3 + km24 / 6)
        m2 += dm2

        dp2 = (kp21 / 6 + kp22 / 3 + kp23 / 3 + kp24 / 6)
        p2 += dp2

        if p1 < 0 or p2 < 0:
            x -= dx
            m1 -= dm1
            p1 -= dp1
            m2 -= dm2
            p2 -= dp2
            dx /= 10

        # Summing masses and pressures
        m_total = m1 + m2
        p_total = p1 + p2

        xvec.append(x)
        m1vec.append(m1)
        p1vec.append(p1)
        m2vec.append(m2)
        p2vec.append(p2)
        m_total_vec.append(m_total)
        p_total_vec.append(p_total)

        x += dx

    # PLOT PRESSURE VS RADIUS FOR BOTH FLUIDS AND TOTAL
    plt.subplot(1, 2, 1)
    plt.plot(xvec, p_total_vec, label='Total Pressure', linestyle='--')
    plt.xlabel("Radius (km)")
    plt.ylabel("Dimensionless Pressure ")
    plt.title("P-r plot")
    plt.legend()

    # PLOT MASS VS RADIUS FOR BOTH FLUIDS AND TOTAL
    plt.subplot(1, 2, 2)
    plt.plot(xvec, m_total_vec, label='Total Mass', linestyle='--')
    plt.xlabel("Radius (km)")
    plt.ylabel("Dimensionless Mass ")
    plt.title("M-r plot")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
