from bosonstar.ComplexBosonStar import Complex_Boson_Star
import matplotlib.pyplot as plt

# =====================
#  All important definitions
# =====================

# Physics defintions
#phi0 = 0.40         # centeral phi
phi_val = [0.01,0.04,0.1,0.4]
#phi0 = 0.5
D = 5.0             # Dimension (total not only spacial)
Lambda = -0.2       # Cosmological constant
# Solver definitions
Rstart = 3
Rend = 50.00
deltaR = 1
N = 100000
e_pow_minus_delta_guess = 0.4999
verbose = 2
eps = 1e-10  # Small epsilon to avoid r \neq 0

# ====================================
#   Main routine
# ====================================
for phi0 in phi_val :
    pewpew = Complex_Boson_Star(e_pow_minus_delta_guess, phi0, D, Lambda, verbose)

    pewpew.print_parameters()


    alpha0 = pewpew.radial_walker(Rstart, Rend, deltaR, N, eps)

    # =====================================
    #   Output and plotting
    # =====================================
    soldict = pewpew.get_solution()

    # Makes sure that lapse goes asymptotically to 1
    # (Not an essential step, but recommended)
    pewpew.normalise_edelta()

    pewpew.check_Einstein_equation()

    # ===============================
    path = pewpew.get_path()
    pewpew.plot_solution()
    pewpew.print_solution()

plt.savefig("phi_values.png")