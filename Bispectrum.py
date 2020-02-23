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

    return transfer_splines[ell1]((2 * ell1 + 1) / 2 * x) * transfer_splines[ell2]((2 * ell2 + 1) / 2 * x) * \
           transfer_splines[ell3]((2 * ell3 + 1) / 2 * x) * ((2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) *
                                                             (ell1 + ell2 + ell3) / 2)


def build_grid(ell_sum=4000, ell_cut=1700, ell_step=10):
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

        ell_list = np.arange(10, 1700, 10)

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

        for el1 in ell_list:
            for el2 in ell_list:
                for el3 in ell_list:

                    if el1 + el2 + el3 != 4000:
                        continue

                    result, err = sciint.quad(bispectrum_integrand, -20, -5,
                                              args=(el1, el2, el3, transfer_spline_list),
                                              epsabs=5E-14, epsrel=5E-14, limit=5000)  # previously error of 1E-16

                    ell1_list.append(el1)
                    ell23_list.append(el2 - el3)
                    results_list.append(result)

        print('--- Finished bispectrum integration')
        finish_time = time.time()
        print('--- Bispectrum took ' + str(finish_time - start_time) + ' seconds ---')
        print('--- with an average of ' + str(len(results_list) / (finish_time - start_time)) + ' samples / second ---')

        return ell1_list, ell23_list, results_list


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

    for ell in ell_list:
        transfer_k, transfer_data = transfers[ell]  # boltzmann.get_transfer(ell)
        transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
        transfer_spline_list[ell] = memoize(transfer_spline)

    for index, row in ell_dataframe.iterrows():
        if row['ell1'] + row['ell2'] + row['ell3'] != 4000:
            # Sanity check
            continue

        result, err = sciint.quad(bispectrum_integrand, -20, -5,
                                  args=(row['ell1'], row['ell2'], row['ell3'], transfer_spline_list),
                                  epsabs=5E-12, epsrel=5E-12, limit=5000)  # TODO: check error

        temp = {'ell1': row['ell1'], 'ell2': row['ell2'], 'ell3': row['ell3'], 'value': result}
        result_list.append(temp)

    return result_list
