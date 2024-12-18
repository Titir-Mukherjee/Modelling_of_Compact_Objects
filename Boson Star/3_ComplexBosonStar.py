from scipy.interpolate import interp1d
import os
import time

import numpy as np
import scipy.integrate as spi
import scipy.optimize as opi
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Complex_Boson_Star:

    # ------------------------------------------------------------
    # Physical parameter that are set when initialising the class
    # ------------------------------------------------------------
    e_pow_minus_delta_guess = None
    # central value of the scalar field, important value
    _phi0 = None
    # Dimension of the problem
    _Dim = None
    # Cosmological constant
    _Lambda = None

    # ------------------------------------------------------------
    # Function parameters
    # ------------------------------------------------------------
    verbose = None
    path = None

    # ------------------------------------------------------------
    # Internal variables that will be found after the program ran
    # ------------------------------------------------------------
    _e_pow_minus_delta_final = None
    _omega = None
    __solution_array = None
    __solution_r_pos = None

    _finished_shooting = False

    def __init__(self, e_pow_minus_delta_guess, phi0, Dim, Lambda, verbose=0,
            rtol = 1e-10, atol = 1e-10):

        self.e_pow_minus_delta_guess = e_pow_minus_delta_guess
        self._phi0 = phi0
        self._Dim = Dim
        self._Lambda = Lambda
        self._atol = atol
        self._rtol = rtol

        # Will give more messages with increasing value
        self.verbose = verbose

        self.make_file()
        return None

    def print_parameters(self):
        print("----------------------------------------------------")
        print((r"The cosmological constant $\Lambda$ ", self._Lambda))
        print(("The dimension of the problen        ", self._Dim))
        print((r"Central value of $\phi$             ", self._phi0))
        print(" Please cite https://arxiv.org/abs/gr-qc/0309131    ")
        print("----------------------------------------------------")

    def eqns(self, y, r):
        """ Differential equation for scalar fields from arXiv:gr-qc/0309131

        Parameters:
            y (list with reals): current status vector ( a(r), m(r), phi(r), pi(r) )
            r (real) : current position

        Returns:
            dydr (list with reals): derivative for y in r
        """
        D = float(self._Dim)
        Lambda = self._Lambda
        e_pow_minus_delta, m, phi, pi = y  # Unpack the state variables from y
        
        # Where e_pow_minus_delta  = e^{-\delta}
        F = (1 - 2 * m / r**(D - 3) - 2 * Lambda * r**2 / ((D - 2) * (D - 1)))

        # Compute the derivatives
        de_pow_minus_deltadr = r * (e_pow_minus_delta * pi**2.0 + e_pow_minus_delta**(-1) * phi**2 / F**2)
        dmdr = r**(D - 2) * 0.5 * (F * pi**2 + 2 * phi**2 + e_pow_minus_delta**(-2) * phi**2 / F)
        dphidr = pi

        # Ensure all returned elements are scalars
        dpidr = -(phi / (e_pow_minus_delta**2 * F**2)) + 2 * phi / F - (de_pow_minus_deltadr * pi) / e_pow_minus_delta - (2 * pi) / r - (D * pi) / r

        # Return the derivatives as a list of 4 scalars
        dydr = [de_pow_minus_deltadr, dmdr, dphidr, dpidr]
        
        return dydr


    def shoot(self, e_pow_minus_delta_at_zero, r, output=False):
        """ Solves differential equation
        Parameters:
            e_pow_minus_delta_at_zero (real): The lapse value guess at r = rmin
            r       (real array) : Radial points used for solver
            output  (bool)       : if True outputs whole solution array
        Returns:
            phi_end (real): The phi value at r = rmax
            or
            sol     (real array) : array containing solution
        """
        # Ensure proper initial conditions based on system equations
        y0 = np.array([e_pow_minus_delta_at_zero, 0.0, self._phi0, 0.0], dtype="object")  # Ensure dtype="object"
        # Solve the system of ODEs using odeint
        sol = spi.odeint(self.eqns, y0, r, atol=self._atol, rtol=self._rtol)
        phi_end = sol[-1, 2]  # Final value of phi

        if not output:
            return phi_end
        else:
            return sol


    def radial_walker(self, r_start, r_end, delta_R, N, eps):
        """ Performs shooting for multiple radii rmax shooting process.

        Parameters:
            r_start (real) : first rmax for which shooting is performed
            r_end (real) : maximum rmax for which shooting is performed
            delta_R (real) : step size
            N (real) : number of grid points

        Returns:
            alpha0 (real): alpha0 for rmax
        """
        range_list = np.arange(r_start, r_end, delta_R)
        e_pow_minus_delta_guess_tmp = np.array([self.e_pow_minus_delta_guess])  # Ensure scalar or array format

        if self.verbose >= 1:
            print("Shooting started")
        if self.verbose >= 1:
            start = time.time()

        for R_max in range_list:
            r = np.linspace(eps, R_max, N)

            # Ensure 'fun' returns a scalar or a numpy array with correct shape
            def fun(x): 
                return np.array([self.shoot(x, r)])  # Wrap the result in a numpy array

            # Use root-finding with the correct shape for the guess
            root = opi.root(fun, e_pow_minus_delta_guess_tmp)

            # Update the guess for the next iteration
            e_pow_minus_delta_guess_tmp = root.x

            if self.verbose >= 2:
                print((
                    "Edelta at R = eps ",
                    e_pow_minus_delta_guess_tmp[0],
                    " with Rmax ",
                    R_max))

        if self.verbose >= 1:
            print(("Shooting finished in ", time.time() - start, "sec"))

        self._finished_shooting = True
        output_solution = True
        self.__solution_r_pos = np.linspace(eps, r_end, N)
        self.__solution_array = self.shoot(
            e_pow_minus_delta_guess_tmp[0],
            self.__solution_r_pos,
            output_solution)
        self._e_pow_minus_delta_final = e_pow_minus_delta_guess_tmp

        return e_pow_minus_delta_guess_tmp[0]


    def check_Einstein_equation(self):
        """ Checks if Einstein equation is fulfilled for D = 5
        Returns:
            Error (array(4,N)) : Array of Einstein Violation tt rr phiphi
            thetatheta
        """

        if self.__solution_array is None or self.__solution_r_pos is None:
            print("----------------------------------------")
            print("WARNING: SHOOTING HAS NOT BEEN PERFORMED")
            print("----------------------------------------")
            return None

        if (self._Dim != 5):
            print("----------------------------------------")
            print("CHECK ONLY VALID FOR D = 5 !")
            print("----------------------------------------")
            return None

        pi = self.__solution_array[:, 3]
        phi = self.__solution_array[:, 2]
        m = self.__solution_array[:, 1]
        e_pow_minus_delta = self.__solution_array[:, 0]
        r = self.__solution_r_pos
        dr = r[1] - r[0]
        dmdr = 1.0 / dr * np.gradient(m)
        d2mdr2 = 1.0 / dr * np.gradient(dmdr)
        de_pow_minus_deltadr = 1.0 / dr * np.gradient(e_pow_minus_delta)
        d2e_pow_minus_deltadr2 = 1.0 / dr * np.gradient(de_pow_minus_deltadr)

        if self._omega is None:
            omega = 1
        else:
            omega = self._omega

        # A random non-trivial angle for the 4th spatial dimension
        theta = 0.71223

        Lambda = self._Lambda

        Einstein_tt = -(e_pow_minus_delta**2 * phi**2) / 2. - (omega * phi**2) / 4. - (e_pow_minus_delta**2 * pi**2) / 4. - (e_pow_minus_delta**2 * Lambda * m * pi**2) / 6. - (dmdr * e_pow_minus_delta**2 * m) / r**5 - (e_pow_minus_delta**2 * m**2 * pi**2) / r**4 + (dmdr * e_pow_minus_delta**2) / (2. * r**3) + (
            e_pow_minus_delta**2 * m * phi**2) / r**2 + (e_pow_minus_delta**2 * m * pi**2) / r**2 - (dmdr * e_pow_minus_delta**2 * Lambda) / (12. * r) + (e_pow_minus_delta**2 * Lambda * phi**2 * r**2) / 12. + (e_pow_minus_delta**2 * Lambda * pi**2 * r**2) / 12. - (e_pow_minus_delta**2 * Lambda**2 * pi**2 * r**4) / 144.

        Einstein_rr = phi**2 / (2. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - (omega**2 * phi**2) / (4. * e_pow_minus_delta**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - pi**2 / (4. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - (Lambda * m * pi**2) / (6. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) + (dmdr * m) / (r**5 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) + (2 * de_pow_minus_deltadr * m**2) / (e_pow_minus_delta * r**5 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - (m**2 * pi**2) / (r**4 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - dmdr / (2. * r**3 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - (2 * de_pow_minus_deltadr * m) / (e_pow_minus_delta * r**3 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - (m * phi**2) / (r**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) + (m * pi**2) / (
            r**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) + de_pow_minus_deltadr / (2. * e_pow_minus_delta * r * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) + (dmdr * Lambda) / (12. * r * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) + (de_pow_minus_deltadr * Lambda * m) / (3. * e_pow_minus_delta * r * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - (de_pow_minus_deltadr * Lambda * r) / (6. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - (Lambda * phi**2 * r**2) / (12. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) + (Lambda * pi**2 * r**2) / (12. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) + (de_pow_minus_deltadr * Lambda**2 * r**3) / (72. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2) - (Lambda**2 * pi**2 * r**4) / (144. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)**2)

        Einstein_phiphi = -d2mdr2 / 6. - (de_pow_minus_deltadr * dmdr) / (2. * e_pow_minus_delta) - (d2e_pow_minus_deltadr2 * m) / (3. * e_pow_minus_delta) - (m * pi**2) / 2. + (de_pow_minus_deltadr * m) / (3. * e_pow_minus_delta * r) + (de_pow_minus_deltadr * r) / (3. * e_pow_minus_delta) + (d2e_pow_minus_deltadr2 * r**2) / (6. * e_pow_minus_delta) + (
            phi**2 * r**2) / 2. + (pi**2 * r**2) / 4. - (5 * de_pow_minus_deltadr * Lambda * r**3) / (36. * e_pow_minus_delta) - (d2e_pow_minus_deltadr2 * Lambda * r**4) / (36. * e_pow_minus_delta) - (Lambda * pi**2 * r**4) / 24. - (omega**2 * phi**2 * r**2) / (4. * e_pow_minus_delta**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.))

        Einstein_thetatheta = -(d2mdr2 * np.sin(theta)**2) / (6. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (de_pow_minus_deltadr * dmdr * np.sin(theta)**2) / (2. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (2 * d2e_pow_minus_deltadr2 * m * np.sin(theta)**2) / (3. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (m * phi**2 * np.sin(theta)**2) / (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.) - (m * pi**2 * np.sin(theta)**2) / (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.) - (2 * de_pow_minus_deltadr * m**2 * np.sin(theta)**2) / (3. * e_pow_minus_delta * r**3 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (d2mdr2 * m * np.sin(theta)**2) / (3. * r**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (de_pow_minus_deltadr * dmdr * m * np.sin(theta)**2) / (e_pow_minus_delta * r**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (2 * d2e_pow_minus_deltadr2 * m**2 * np.sin(theta)**2) / (3. * e_pow_minus_delta * r**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (m**2 * pi**2 * np.sin(theta)**2) / (r**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (de_pow_minus_deltadr * m * np.sin(theta)**2) / (3. * e_pow_minus_delta * r * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (de_pow_minus_deltadr * r * np.sin(theta)**2) / (3. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (2 * de_pow_minus_deltadr * Lambda * m * r * np.sin(theta)**2) / (9. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (d2e_pow_minus_deltadr2 * r**2 * np.sin(theta)**2) / (6. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (
            d2mdr2 * Lambda * r**2 * np.sin(theta)**2) / (36. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (de_pow_minus_deltadr * dmdr * Lambda * r**2 * np.sin(theta)**2) / (12. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (d2e_pow_minus_deltadr2 * Lambda * m * r**2 * np.sin(theta)**2) / (9. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (phi**2 * r**2 * np.sin(theta)**2) / (2. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (omega**2 * phi**2 * r**2 * np.sin(theta)**2) / (4. * e_pow_minus_delta**2 * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (pi**2 * r**2 * np.sin(theta)**2) / (4. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (Lambda * m * pi**2 * r**2 * np.sin(theta)**2) / (6. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (7 * de_pow_minus_deltadr * Lambda * r**3 * np.sin(theta)**2) / (36. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (d2e_pow_minus_deltadr2 * Lambda * r**4 * np.sin(theta)**2) / (18. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (Lambda * phi**2 * r**4 * np.sin(theta)**2) / (12. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) - (Lambda * pi**2 * r**4 * np.sin(theta)**2) / (12. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (5 * de_pow_minus_deltadr * Lambda**2 * r**5 * np.sin(theta)**2) / (216. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (d2e_pow_minus_deltadr2 * Lambda**2 * r**6 * np.sin(theta)**2) / (216. * e_pow_minus_delta * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.)) + (Lambda**2 * pi**2 * r**6 * np.sin(theta)**2) / (144. * (1 - (2 * m) / r**2 - (Lambda * r**2) / 6.))

        plt.figure(figsize=(10, 10))
        plt.plot(r, np.abs(Einstein_tt), label="G_tt-T_tt")
        plt.plot(r, np.abs(Einstein_rr), label="G_rr-T_rr")
        plt.plot(r, np.abs(Einstein_phiphi), label="G_phph-T_phph")
        plt.plot(r, np.abs(Einstein_thetatheta), label="G_thth-T_thth")
        plt.yscale('log')
        plt.ylim(top=1.0)
        plt.legend()
        plt.savefig(self.path + "/EinsteinError.png")

        return np.array([Einstein_tt, Einstein_rr,
                         Einstein_phiphi, Einstein_thetatheta])

    def normalise_edelta(self):
        """ Extracts omega for e_pow_delta by the coordinate transformation  t -> omega t

        """
        if self._omega is None:
            omega = self.__solution_array[-1, 0]
            self._omega = omega
            self.__solution_array[:, 0] *= omega
        else:
            print(" edelta has been already normalised ")

    def make_file(self):
        """ Creates Folder for current physics problem if they do not yet exist
        """

        name_Field = "scalar_field_star"
        name_Lambda = "/Lambda_" + str(self._Lambda)
        name_Dim = "/Dim_" + str(self._Dim)
        name_Param = "/phi0_" + str(self._phi0)

        path = name_Field
        if not os.path.exists(path):
            os.mkdir(path)
        path += name_Lambda
        if not os.path.exists(path):
            os.mkdir(path)
        path += name_Dim
        if not os.path.exists(path):
            os.mkdir(path)
        path += name_Param
        if not os.path.exists(path):
            os.mkdir(path)
            if self.verbose >= 1:
                print(("Create Folder with relative", path, "."))
        else:
            if self.verbose >= 1:
                print(("Folder with path", path, "already exists."))

        self.path = path

    def get_path(self):
        """ return
              path (string): Relative path used for outputs
        """
        if self.path is None:
            self.make_file()
        return self.path

    def get_solution(self):
        """return
             solution_array (dictonary) : solution array for Rmax
        """
        if self.__solution_array is None or self.__solution_r_pos is None:
            print("----------------------------------------")
            print("WARNING: SHOOTING HAS NOT BEEN PERFORMED")
            print("----------------------------------------")
            return None
        else:
            soldict = {
                "rpos": self.__solution_r_pos,
                "phi": self.__solution_array[:, 2],
                "m": self.__solution_array[:, 1],
                "e_pow_minus_delta": self.__solution_array[:, 0]
            }
            return soldict

    def print_solution(self):
        """ Prints solution if shooting has been performed already

        """
        if self.path is None:
            self.make_file()
        if self.__solution_array is None or self.__solution_r_pos is None:
            print("----------------------------------------")
            print("WARNING: SHOOTING HAS NOT BEEN PERFORMED")
            print("----------------------------------------")
        else:
            if self.path is None:
                self.make_file()
            if self._omega is None:
                self.normalise_edelta()
            phi = self.__solution_array[:, 2]
            m = self.__solution_array[:, 1]
            e_pow_delta = 1 / self.__solution_array[:, 0]
            r = self.__solution_r_pos
            omega = self._omega

            np.savetxt(self.path + "/omega.dat", [omega])
            np.savetxt(self.path + "/rvals.dat", r)
            np.savetxt(self.path + "/edelta.dat", e_pow_delta)
            np.savetxt(self.path + "/m.dat", m)
            np.savetxt(self.path + "/phi.dat", phi)

    def plot_solution(self):
        """ Prints solution if shooting has been performed already

        """
        if self.path is None:
            self.make_file()
        if self.__solution_array is None or self.__solution_r_pos is None:
            print("----------------------------------------")
            print("WARNING: SHOOTING HAS NOT BEEN PERFORMED")
            print("----------------------------------------")
        else:

            if self.verbose >= 1:
                print("Plotting started")
            if self.verbose >= 1:
                start = time.time()

            phi = self.__solution_array[:, 2]
            m = self.__solution_array[:, 1]
            e_pow_delta = 1 / self.__solution_array[:, 0]
            r = self.__solution_r_pos

            # find 90 % radius of R
            Rguess = 0.01
            maxphi = max(phi)
            phi_tmp_fun = interp1d(r, phi - maxphi * 0.1)
            root = opi.root(phi_tmp_fun, Rguess)
            R90 = root.x[0]

            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 10))
            ax1.plot(r, e_pow_delta, 'b', )
            ax2.plot(r, m, 'g')
            ax3.plot(r, phi, 'r')

            ax3.set_xlabel('t')

            ax1.set_ylabel(r'$ e^{\delta (t)}$')
            ax2.set_ylabel('$ m (t)$')
            ax3.set_ylabel(r'$\phi (t)$')

            ax1.set_xlim([0, R90 * 2])
            ax2.set_xlim([0, R90 * 2])
            ax3.set_xlim([0, R90 * 2])

            ax1.grid()
            ax2.grid()
            ax3.grid()

            plt.savefig(self.path + "/overview_scalar.png")

            if self.verbose >= 1:
                print(("Plotting finished in ", time.time() - start, " sec"))
