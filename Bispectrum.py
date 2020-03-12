# Created by Alessandro Maraio on 22/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is dedicated to performing the three-point function integrals.
It will be expected that a lot of the details of the calculations will be abstracted away
from the user. Perhaps a few different integral methods could be used, which allows for cross-comparison
to ensure numerical accuracy and stability with speed comparisons too.
"""

import time
import numpy as np
import pandas as pd
from scipy import interpolate as interp
from scipy import integrate as sciint


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


def bispectrum_integrand(x, ell1, ell2, ell3, transfer_splines):
    # Defines the integrand of the bispectrum as a function of x, and additionally ell1, ell2, ell3 and
    # transfer function data, with two-point function data to be added later.
    # Note: Here we have defined the integrand in log k-space as it makes the transfer functions slightly
    # nicer to evaluate and so the integral is better behaved and more accurate.
    # Also note: We include the conventional normalisation (Fergusson) factor  here, as that way all values are about
    # the same order of magnitude, so that way a relative error makes more sense.

    x = np.exp(x)

    integrand = transfer_splines[ell1]((2 * ell1 + 1) / (2 * x)) * transfer_splines[ell2]((2 * ell2 + 1) / (2 * x)) * \
                transfer_splines[ell3]((2 * ell3 + 1) / (2 * x))

    integrand *= 8 * np.sqrt(1 / (np.pi ** 3 * (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1)))

    integrand *= (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) * (ell1 + ell2 + ell3) / 2

    return integrand

    # return transfer_splines[ell1]((2 * ell1 + 1) / (2 * x)) * transfer_splines[ell2]((2 * ell2 + 1) / (2 * x)) * \
    #       transfer_splines[ell3]((2 * ell3 + 1) / (2 * x)) * \
    #       ((2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) * (ell1 + ell2 + ell3) / 2)


def equal_ell_integrand(x, ell, transfer_splines):
    # Defines the integrand of the bispectrum as a function of x, and additionally ell
    # transfer function data, with two-point function data to be added later.
    # Note: Here we have defined the integrand in log k-space as it makes the transfer functions slightly
    # nicer to evaluate and so the integral is better behaved and more accurate.
    # Also note: We include the conventional normalisation (Fergusson) factor  here, as that way all values are about
    # the same order of magnitude, so that way a relative error makes more sense.

    x = np.exp(x)

    integrand = transfer_splines[ell]((ell + 0.5) / x) ** 3

    integrand *= (2 / np.pi) ** 3 * (np.pi / (2 * ell + 1)) ** 1.5

    integrand *= (2 * ell + 1) ** 3 * ell * (3 * ell + 1) / (2 * ell + 1)

    return integrand


def build_grid_ell_sum(ell_sum=4000, ell_cut=1700, ell_step=10):
    """
    Builds a grid of allowed [ell1, ell2, ell3] values that sum to produce ell_sum, with an individual maximum
    ell value of ell_cut, built using steps of ell_step in the individual values.

    Returns pandas dataframe of [ell1, ell2, ell3].
    """

    print('--- Building grid of ells ---', flush=True)

    ell_list = np.arange(10, ell_cut, ell_step)

    rows = []

    for ell1 in ell_list:
        for ell2 in ell_list:
            for ell3 in ell_list:
                if ell1 + ell2 + ell3 == ell_sum:
                    temp = {'ell1': ell1, 'ell2': ell2, 'ell3': ell3}
                    rows.append(temp)

    data = pd.DataFrame(rows)

    print('--- Built grid of ells ---', flush=True)
    print('--- Number of ell configurations ' + str(len(rows)) + ' ---', flush=True)
    return data


def build_ell_grid(ell_step=10, ell_max=2000):
    """
    Builds a grid of allowed (ell1, ell2, ell3) values that are allowed by the ell selection rules in place
    for the bispectrum configurations. We build the grid in steps using ell_steps and up to an individual
    ell maximum of ell_max.

    The selection rules are:
        - Parity condition: ell1 + ell2 + ell3 = even
        - Triangle condition: ell1, ell2, ell3 must form a triangle from their values
    """

    print('--- Building grid of ells ---', flush=True)

    ell_list = np.arange(10, ell_max, ell_step)

    allowed_ells = []

    for ell1 in ell_list:
        for ell2 in ell_list:
            for ell3 in ell_list:
                if (ell1 + ell2 + ell3) % 2 != 0:
                    continue

                if (ell1 + ell2 <= ell3) or (ell1 + ell3 <= ell2) or (ell2 + ell3 <= ell1):
                    continue

                allowed_ells.append({'index': len(allowed_ells), 'ell1': ell1, 'ell2': ell2, 'ell3': ell3})

    allowed_ells = pd.DataFrame(allowed_ells)

    print('--- Built grid of ells ---', flush=True)
    print('--- Number of ell configurations ' + str(allowed_ells.shape[0]) + ' ---', flush=True)

    return allowed_ells


class Bispectrum:
    def __init__(self, transfer, database):
        # Set up the class, recording the database and transfer functions that will be used in the integration
        self.transfer = transfer
        self.database = database
        # * self.type = self.database.type

    def integrate(self):
        # Normal, single threaded CMB integration.
        start_time = time.time()
        print('--- Starting bispectrum integration ---')

        ell_list = np.arange(10, 1600, 10)

        transfer_spline_list = {}

        for ell in ell_list:
            transfer_k, transfer_data = self.transfer.get_transfer(ell)
            transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
            memorize_spline = memoize(transfer_spline)
            transfer_spline_list[ell] = memorize_spline

        # x_list = np.linspace(-25, 25, num=100000)
        # for x in x_list:
        # plt.semilogy(x_list, np.abs(bispectrum_integrand(x_list, 1030, 1400, 1490, transfer_spline_list)), 'b')
        # plt.show()

        ell1_list = []
        ell23_list = []
        results_list = []

        quad = sciint.quad

        for el1 in ell_list:
            for el2 in ell_list:
                for el3 in ell_list:

                    if el1 + el2 + el3 != 4000:
                        continue

                    result, err = quad(bispectrum_integrand, -17, 250,
                                       args=(el1, el2, el3, transfer_spline_list),
                                       epsabs=1E-6, epsrel=1E-6, limit=5000)  # previously error of 1E-16

                    ell1_list.append(el1)
                    ell23_list.append(el2 - el3)
                    results_list.append(result)

        print('--- Finished bispectrum integration')
        finish_time = time.time()
        print('--- Bispectrum took ' + str(round(finish_time - start_time, 2)) + ' seconds ---')
        print('--- with an average of ' + str(round(len(results_list) / (finish_time - start_time), 2))
              + ' samples / second ---')
        print('--- for integrating ' + str(len(results_list)) + ' samples ---')

        return ell1_list, ell23_list, results_list

    def integrate_constant_ell(self, ell_max=2000, ell_step=5):
        """
        Integrates the same-ell CMB bispectrum, which is where ell1 = ell2 = ell3 = ell.
        This allows for conventional plots to be made for the value of the bispectrum, instead of 3D or 3D isosurface
        plots.

        Takes in arguments of a ell_max, which is the maximum value of ell that will be integrated up to, and
        ell_step which is the step length between ell sampling points.

        Returns a list of dictionaries of each integration with data [ell, value]
        """
        start_time = time.time()
        print('--- Starting bispectrum integration ---')

        result_list = []

        ell_list = np.arange(30, ell_max, ell_step)

        transfer_spline_list = {}

        for ell in ell_list:
            transfer_k, transfer_data = self.transfer.get_transfer(ell)
            transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
            transfer_spline_list[ell] = memoize(transfer_spline)

        quad = sciint.quad

        for ell in ell_list:
            result, err = quad(equal_ell_integrand, 0, 20,
                               args=(ell, transfer_spline_list),
                               epsabs=1E-12, epsrel=1E-12, limit=5000)  # TODO: check error

            # result *= np.sqrt(8 / (np.pi ** 3 * (ell + 0.5) ** 3))

            temp = {'ell': ell, 'value': result}
            result_list.append(temp)

        print('--- Finished same-ell bispectrum integration')
        finish_time = time.time()
        print('--- Bispectrum took ' + str(round(finish_time - start_time, 2)) + ' seconds ---')
        print('--- with an average of ' + str(round(len(result_list) / (finish_time - start_time), 2))
              + ' samples / second ---')
        print('--- for integrating ' + str(len(result_list)) + ' samples ---')

        return result_list


def parallel_integrate(ell_dataframe, transfers):
    """
    Parallel approach of computing the CMB bispectrum.

    Takes in arguments of a dataframe of ell values that we will be integrating, and a dictionary of
    transfer functions that are indexed by their ell value, with each entry a list of
    [transfer_k, transfer_data] values.

    Returns a list of dictionaries of each integration with data [ell1, ell2, ell3, value]
    """
    result_list = []

    ell_list = np.unique(np.concatenate((ell_dataframe['ell1'], ell_dataframe['ell2'], ell_dataframe['ell3'])))

    transfer_spline_list = {}

    quad = sciint.quad

    for ell in ell_list:
        transfer_k, transfer_data = transfers[ell]  # boltzmann.get_transfer(ell)
        transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
        transfer_spline_list[ell] = memoize(transfer_spline)

    for index, row in ell_dataframe.iterrows():
        # ! if row['ell1'] + row['ell2'] + row['ell3'] != 4000:
        # Sanity check
        # !    continue

        result, err = quad(bispectrum_integrand, 2, 15,
                           args=(row['ell1'], row['ell2'], row['ell3'], transfer_spline_list),
                           epsabs=1E-6, epsrel=1E-6, limit=5000)  # TODO: check error

        temp = {'index': row['index'], 'ell1': row['ell1'], 'ell2': row['ell2'], 'ell3': row['ell3'], 'value': result}
        result_list.append(temp)

    return result_list
