# Created by Alessandro Maraio on 02/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is dedicated to performing the two-point function integrals.
It will be expected that a lot of the details of the calculations will be abstracted away
from the user. Perhaps a few different integral methods could be used, which allows for cross-comparison
to ensure numerical accuracy and stability with speed comparisons too.
"""


# Import required modules for integration
import time
import numpy as np
import pandas as pd
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
        result *= 2.725 ** 2  # Use correct units of (mu K)^2 for the Cl's

    else:
        # Integrate using only the transfer function data points.
        # The integral is then calculated using the Simpson method for sampled data-sets.
        integrand = []
        transfer_k = np.log(transfer_k)
        for k_itter, transfer_data_itter in zip(transfer_k, transfer_data):
            integrand.append(
                2 * transfer_data_itter * transfer_data_itter * twopf_spline(np.exp(k_itter)) * 1E12 * el * (el + 1))

        result = sciint.simps(integrand, transfer_k)
        result *= 2.725 ** 2  # Use correct units of (mu K)^2 for the Cl's

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
        # Set up the class, recording the database and transfer functions that will be used in the integration
        self.transfer = transfer
        self.database = database
        self.type = self.database.type

    def integrate_power_spectrum(self, use_splines=False, parallel=False):
        """
        Function that integrates the CMB power spectrum, given the transfer function and inflationary power spectrum

        Args:
            use_splines (bool): Whether to use splines in the integration or not. By default, we use a sampled-based
                integration routine, and so we do not need to use splines. However, if we enable this, then we go to a
                functional-based integral solver using splines instead.
            parallel (bool): Whether we should use a parallel-based approach. By doing so, we get the speed increase
                of using multiple threads, however this disables the inflation power spectrum memoization, which
                decreases speed significantly.

        Returns:
            Two lists:
                - List of ell values at which the power spectrum is evaluated at
                - List of values of the power spectrum at these ell values.
        """

        # Check that the database type is for a two-point function run and so can integrate the power spectrum
        if self.type != 'twopf':
            raise RuntimeError('Can not integrate the power spectrum on integration type that is not twopf.')

        ell_list = self.transfer.get_ell_list()
        c_ell_list = []

        print('--- Starting two-point function integral ---')

        # Get the provided two-point function data
        twopf_dataframe = self.database.get_dataframe
        twopf_data = twopf_dataframe['twopf']

        # Get the physical k values that are provided for the above twopf data
        k_data = pd.read_csv(str(self.database.k_table), sep='\t')

        # Ensure that each twopf value corresponds to each physical k value
        if twopf_data.shape[0] != k_data.shape[0]:
            raise RuntimeError('The length of the provided two-point function database and k_table are not equal, '
                               'and so were not formed as part of the same task. Please ensure they were generated '
                               'at the same time.')

        # Spline the twopf data over the physical k values
        twopf_spline = interp.CubicSpline(k_data['k_physical'], twopf_data)

        # If we are not using parallelization here, then we can use function memorisation on the twopf spline
        # to help increase the speed of the integration
        if not parallel:
            twopf_spline = memoize(twopf_spline)

        # Keep track on how long the integral takes using different methods.
        start_time = time.time()

        if not parallel:

            for index, ell in enumerate(ell_list):
                transfer_k, transfer_data = self.transfer.get_transfer(ell)

                if use_splines:
                    # Build a spline out of the transfer function. Very important that we have it sent to return zero
                    # for values outside the interpolated region, otherwise this induces large numerical errors.
                    # TODO: compare with other spline methods and/or libraries to see if performance and/or accuracy
                    #  can be improved
                    transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
                    result, err = sciint.quad(log_integrand, -15, 4, args=(ell, transfer_spline, twopf_spline),
                                              epsabs=1E-10, epsrel=1E-10, limit=5000)
                    result *= 2.725**2  # Use correct units of (mu K)^2 for the Cl's

                else:
                    # If we are not using the transfer function splines, then use the points provided by CAMB
                    # by which the transfer functions are evaluated at, then evaluate the rest of the integrand
                    # at this point, and then use basic Simpson's rule integration to evaluate it.
                    # This offers a significant speed improvement over using splines

                    integrand_list = []
                    transfer_k = np.log(transfer_k)  # Go into log space, which makes integral much nicer

                    # Go through the points at which the transfer functions are evaluated at, and evaluate
                    # the rest of the integrand
                    for k_itter, transfer_data_itter in zip(transfer_k, transfer_data):
                        integrand_list.append(2 * transfer_data_itter * transfer_data_itter *
                                              twopf_spline(np.exp(k_itter)) * 1E12 * ell * (ell + 1))

                    # Call SciPy Simpson's rule integration on the integrand
                    result = sciint.simps(integrand_list, transfer_k)
                    result *= 2.725 ** 2  # Use correct units of (mu K)^2 for the Cl's

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
        print(' -- Time taken was ' + str(round(end_time - start_time, 2)) + ' seconds ---')

        return ell_list, c_ell_list

    def integrate_power_spectrum_cpp(self):
        """
        Function to use the C++ routine for the power spectrum integration.

        This should offer a significant speed improvement by using the ctypes library, however it appears to
        not offer such improvements as imagined.
        """
        ell_list = self.transfer.get_ell_list()
        c_ell_list = []

        start_time = time.time()

        # Import required modules for integration using the C++ code
        import os
        import ctypes
        from scipy import LowLevelCallable

        # Use the ctypes library to initialise the functions correctly
        lib = ctypes.CDLL(os.path.abspath('../cpp/build/libCppIntegrand.so'))
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

            result, err = sciint.quad(func, -10, 1, epsabs=1E-18, epsrel=1E-18, limit=5000)

            # result *= ell * (ell + 1) / (2 * np.pi)  # Multiply by common factor when using C_ell values
            c_ell_list.append(result)

        print('--- Finished two-point function integral ---')
        end_time = time.time()
        print('Time taken = ' + str(end_time - start_time))

        return ell_list, c_ell_list
