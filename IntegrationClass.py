# Created by Alessandro Maraio on 02/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is dedicated to performing the two- and three-point function integrals.
It will be expected that a lot of the details of the calculations will be abstracted away
from the user. Perhaps a few different integral methods could be used, which allows for cross-comparison
to ensure numerical accuracy and stability with speed comparisons too.
"""

# Import required modules for integration
import time
import numpy as np
from scipy import interpolate as interp
from scipy import integrate as sciint


def log_integrand(k, ell, trans_spline, tpf_spline):
    # Defines the integrand of the power spectrum as a function of k, and additionally ell, transfer function data
    # and two-point function data.
    # Note: Here we have defined the integrand in log k-space as it makes the transfer functions slightly
    # nicer to evaluate and so the integral is better behaved and more accurate.
    # Also note: We include the conventional factor of ell(ell+1)/2Pi here, as that way all values are about
    # the same order of magnitude, so that way a relative error makes more sense.

    return 2 * (trans_spline(np.exp(k)) ** 2) * tpf_spline(np.exp(k)) * 1E12 * ell * (ell + 1)


def integrate(args):
    """
    This is the function that gets called when wanting to do a parallel power spectrum integral.
    The map function from the multiprocessing library will use this function to iterate through the samples.

    This has the option to either use a spline-based integral, or use only the provided data points for the transfer
    functions, and use a sampled-based integration routine for this.
    In my experience, using a sampled-based integration routine offers a significant speed improvement
    (as the bottleneck in evaluating the splines is eliminated) at negligible accuracy loss.
    """

    el = args[0]
    transfer_k = args[1]
    transfer_data = args[2]
    twopf_spline = args[3]
    use_splines = args[4]

    if use_splines:
        # Integrate using splines and then call the quad library
        transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
        result, err = sciint.quad(log_integrand, -15, 4, args=(args[0], transfer_spline, twopf_spline), epsabs=1E-4,
                                  epsrel=1E-4, limit=5000)

    else:
        # Integrate using only the transfer function data points.
        # The integral is then calculated using the Simpson method for sampled data-sets.
        integrand = []
        transfer_k = np.log(transfer_k)
        for k_itter, transfer_data_itter in zip(transfer_k, transfer_data):
            integrand.append(
                2 * transfer_data_itter * transfer_data_itter * twopf_spline(np.exp(k_itter)) * 1E12 * el * (el + 1))

        result = sciint.simps(integrand, transfer_k)

    return result


def memoize(func):
    """
    Since the power spectrum spline is the same for each ell value, there is no need to repeatedly calculate
    its value when computing the integral. If we 'memorize' the output by saving it in a dictionary, we can
    introduce *significant* speed improvements.

    Note: It is currently not possible to use both memorization and multiprocessing to evaluate the power spectrum.
    It would be good to have this as then it would offer an even greater speed increase.
    """

    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func


class Integration:
    def __init__(self, transfer, database):
        # Set up the class, recording the database and transfer functions that will be used in the intergation
        self.transfer = transfer
        self.database = database
        self.type = self.database.type

    def integrate_power_spectrum(self, use_splines=False, parallel=False):
        # Check that the database type is for a two-point function run and so can integrate the power spectrum
        if self.type != 'twopf':
            raise RuntimeError('Can not integrate the power spectrum on integration type that is not twopf.')

        ell_list = self.transfer.get_ell_list()
        c_ell_list = []

        print('--- Starting two-point function integral ---')
        twopf_data = self.database.get_data()
        twopf_k_range = np.logspace(-6, 1.7, num=401)
        twopf_spline = interp.CubicSpline(twopf_k_range, twopf_data)

        if not parallel:
            twopf_spline = memoize(twopf_spline)

        # Build our integrand function for the SciPy integration.
        # Note: we normalise by 1E12 (which corresponds to units of micro-K^2) here to ensure that the
        # integrand values are not too tiny to cause numerical issues.
        # def integrand(k, trans_spline, tpf_spline):
        #    return (4 * np.pi) * (1 / k) * (trans_spline(k) ** 2) * tpf_spline(k) * 1E12

        def ode_integrand(k, c_ell,  trans_spline, tpf_spline):
            return [(4 * np.pi) * (1 / k) * (trans_spline(k) ** 2) * tpf_spline(k) * 1E12]

        # Keep track on how long the integral takes using different methods.
        start_time = time.time()

        if not parallel:

            for index, ell in enumerate(ell_list):
                transfer_k, transfer_data = self.transfer.get_transfer(index)

                if use_splines:
                    # Build a spline out of the transfer function. Very important that we have it sent to return zero
                    # for values outside the interpolated region, otherwise this induces large numerical errors.
                    # TODO: compare with other spline methods and/or libraries to see if performance and/or accuracy
                    #  can be improved
                    transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
                    result, err = sciint.quad(log_integrand, -15, 4, args=(ell, transfer_spline, twopf_spline),
                                              epsabs=1E-10, epsrel=1E-10, limit=5000)

                else:
                    integrand_list = []
                    transfer_k = np.log(transfer_k)
                    for k_itter, transfer_data_itter in zip(transfer_k, transfer_data):
                        integrand_list.append(2 * transfer_data_itter * transfer_data_itter *
                                              twopf_spline(np.exp(k_itter)) * 1E12 * ell * (ell + 1))

                    result = sciint.simps(integrand_list, transfer_k)

                # Booleans to set the integration method to be used
                # TODO: extract this away into function/class parameter
                UseScipyOdeint  = False

                if UseScipyOdeint:
                    solution = sciint.solve_ivp(ode_integrand, [1E-6, 1], y0=[0], args=(transfer_spline, twopf_spline),
                                                dense_output=True, method='BDF', atol=1E-10, rtol=1E-10)
                    result = solution.sol(0.8)

                c_ell_list.append(result)

        else:
            import multiprocessing as multi

            big_list = []

            for index, ell in enumerate(ell_list):
                transfer_k, transfer_data = self.transfer.get_transfer(index)
                temp = [ell, transfer_k, transfer_data, twopf_spline, use_splines]
                big_list.append(temp)

            pool = multi.Pool(multi.cpu_count())

            c_ell_list = pool.map(integrate, big_list)

        print('--- Finished two-point function integral ---')
        end_time = time.time()
        print('Time taken = ' + str(end_time - start_time))

        return ell_list, c_ell_list

    def integrate_power_spectrum_cpp(self):
        ell_list = self.transfer.get_ell_list()
        c_ell_list = []

        start_time = time.time()

        # Import required modules for integration
        import os
        import ctypes
        from scipy import LowLevelCallable

        lib = ctypes.CDLL(os.path.abspath('./cpp/build/libCppIntegrand.so'))
        lib.get_twopf_data.restype = ctypes.c_double
        lib.get_transfer_data.restype = ctypes.c_double
        lib.get_integrand.restype = ctypes.c_double
        lib.get_integrand.argtypes = [ctypes.c_double]
        lib.get_log_integrand.restype = ctypes.c_double
        lib.get_log_integrand.argtypes = [ctypes.c_double, ctypes.c_void_p]

        print('--- Starting two-point function integral ---')
        twopf_k_range = np.logspace(-6, 1.7, num=401, dtype=float)
        twopf_data = np.array(self.database.get_data(), dtype=float)

        twopf_k_range_ctype = twopf_k_range.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        twopf_data_ctype = twopf_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        lib.set_twopf_data(twopf_k_range_ctype, twopf_data_ctype, len(twopf_data))

        '''
        twopf_list = []
        for k in twopf_k_range:
            p_k = ctypes.c_double(k)
            twopf_list.append(lib.get_twopf_data(p_k))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(twopf_k_range, twopf_data)
        plt.show(block=True)
        '''

        #func = LowLevelCallable(lib.get_log_integrand)

        for index, ell in enumerate(ell_list):
            transfer_k, transfer_data = self.transfer.get_transfer(index)

            transfer_k = np.array(transfer_k, dtype=float)
            transfer_data = np.array(transfer_data, dtype=float)
            transfer_k_ctype = transfer_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            transfer_data_ctype = transfer_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            lib.set_transfer_data(transfer_k_ctype, transfer_data_ctype, len(transfer_k))

            ell_c = ctypes.c_double(ell)
            user_data = ctypes.cast(ctypes.pointer(ell_c), ctypes.c_void_p)

            func = LowLevelCallable(lib.get_log_integrand, user_data)

            k_list = np.logspace(-15, 4, num=8024)
            integrand_list = []
            for k in k_list:
                c_k = ctypes.c_double(k)
                integrand_list.append(lib.get_integrand(c_k))
            """
            import matplotlib.pyplot as plt
            plt.figure()
            plt.loglog(k_list, integrand_list)
            plt.title(r'$\ell$ = ' + str(ell))
            plt.show(block=True)
            """

            # result, err = sciint.quad(func, -10, 1, epsabs=1E-18, epsrel=1E-18, limit=5000)

            # result, err = sciint.nquad(func, [[1E-4, 0.1]], opts={'epsabs': 1E-16, 'epsrel': 1E-16, 'limit': 5000})
            # result = sciint.romb(integrand_list, k_list)

            # result = lib.Do_Cuba_Integration()

            # result *= ell * (ell + 1) / (2 * np.pi)  # Multiply by common factor when using C_ell values
            result = 1
            c_ell_list.append(result)

        print('--- Finished two-point function integral ---')
        end_time = time.time()
        print('Time taken = ' + str(end_time - start_time))

        return ell_list, c_ell_list

