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


def bispectrum_integrand(x, ell1, ell2, ell3, transfer1, transfer2, transfer3, shape_func):
    # Defines the integrand of the bispectrum as a function of x, and additionally ell1, ell2, ell3 and
    # transfer function data, with two-point function data to be added later.
    # Note: Here we have defined the integrand in log k-space as it makes the transfer functions slightly
    # nicer to evaluate and so the integral is better behaved and more accurate.
    # Also note: We include the conventional normalisation (Fergusson) factor  here, as that way all values are about
    # the same order of magnitude, so that way a relative error makes more sense.

    x = np.exp(x)

    integrand = transfer1((ell1 + 0.5) / x) * transfer2((ell2 + 0.5) / x) * transfer3((ell3 + 0.5) / x) * \
                shape_func(np.log10((ell1 + 0.5) / x), np.log10((ell2 + 0.5) / x), np.log10((ell3 + 0.5) / x))

    integrand *= 8 * np.sqrt(1 / (np.pi ** 3 * (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1)))

    integrand *= (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) * (ell1 + ell2 + ell3) / 2

    return integrand

    # return transfer_splines[ell1]((2 * ell1 + 1) / (2 * x)) * transfer_splines[ell2]((2 * ell2 + 1) / (2 * x)) * \
    #       transfer_splines[ell3]((2 * ell3 + 1) / (2 * x)) * \
    #       ((2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) * (ell1 + ell2 + ell3) / 2)


def equal_ell_integrand(x, ell, transfer_spline):
    # Defines the integrand of the bispectrum as a function of x, and additionally ell
    # transfer function data, with two-point function data to be added later.
    # Note: Here we have defined the integrand in log k-space as it makes the transfer functions slightly
    # nicer to evaluate and so the integral is better behaved and more accurate.
    # Also note: We include the conventional normalisation (Fergusson) factor  here, as that way all values are about
    # the same order of magnitude, so that way a relative error makes more sense.

    x = np.exp(x)

    integrand = transfer_spline((ell + 0.5) / x) ** 3

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
                    temp = {'index': len(rows), 'ell1': ell1, 'ell2': ell2, 'ell3': ell3}
                    rows.append(temp)

    data = pd.DataFrame(rows)

    if data.shape[0] == 0:
        raise RuntimeError('The ell grid that was specified has no valid triangle configurations for parameters \n'
                           'ell_sum: ' + str(ell_sum) + ', ell_step: ' + str(ell_step) + ', ell_cut: ' + str(ell_cut) +
                           '\nPlease try again with more relaxed values (decreased ell_step, increased ell_cut)')

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

    if allowed_ells.shape[0] == 0:
        raise RuntimeError('The ell grid that was specified has no valid triangle configurations for parameters \n'
                           'ell_step: ' + str(ell_step) + ', ell_max: ' + str(ell_max) + '\n'
                           'Please try again with more relaxed values (decreased ell_step, increased ell_max)')

    print('--- Built grid of ells ---', flush=True)
    print('--- Number of ell configurations ' + str(allowed_ells.shape[0]) + ' ---', flush=True)

    return allowed_ells


class Bispectrum:
    def __init__(self, transfer, database):
        # Set up the class, recording the database and transfer functions that will be used in the integration
        self.transfer = transfer
        self.database = database
        self.ell_step = None
        self.ell_max = None

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

        Returns a pandas DataFrame which contains the integration data in two columns with values [ell, value]
        """

        # Record the start time, so that way we can calculate the elapsed time for the integrations
        start_time = time.time()
        print('--- Starting bispectrum integration ---')

        # Record the ell step and ell max in the class
        self.ell_step = ell_step
        self.ell_max = ell_max

        # Initiate a blank list which is where the integration results will get put
        result_list = []

        # Create an ell list out of the provided arguments.
        # Note that we start from 30, as this is about the point where the Limber approximation becomes accurate enough
        ell_list = np.arange(30, ell_max, ell_step)

        # Declare local variables instead of referencing global SciPy, a Python performance trick
        quad = sciint.quad
        InterpolatedUnivariateSpline = interp.InterpolatedUnivariateSpline

        for ell in ell_list:
            transfer_k, transfer_data = self.transfer.get_transfer(ell)
            transfer_spline = InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
            # Note: we do not need to use memorization here, as each ell transfer will only be called once

            result, err = quad(equal_ell_integrand, 6, 10,
                               args=(ell, transfer_spline),
                               epsabs=1E-12, epsrel=1E-12, limit=5000)  # TODO: check error

            temp = {'ell': ell, 'value': result, 'err': err}
            result_list.append(temp)

        # Print integration statistics
        print('--- Finished same-ell bispectrum integration')
        finish_time = time.time()
        print('--- Bispectrum took ' + str(round(finish_time - start_time, 2)) + ' seconds ---')
        print('--- with an average of ' + str(round(len(result_list) / (finish_time - start_time), 2))
              + ' samples / second ---')
        print('--- for integrating ' + str(len(result_list)) + ' samples ---')

        # Transform the list of dictionaries to a pandas DataFrame, and then return it
        result_list = pd.DataFrame(result_list)

        return result_list


def parallel_integrate(worker_index, folder, transfers, shape_func):
    """
    Parallel approach of computing the CMB bispectrum.

    Takes in arguments of the worker index, which is the index that this specific core corresponds to in the whole
    MPI core silo, 'save_folder' which is a string corresponding to the location that the worker should read and save
    data to when performing the integration, a dictionary of transfer functions that are indexed by their ell value,
    with each entry a list of [transfer_k, transfer_data] values, and the function 'shape', which is either
    the interpolated inflationary bispectrum or a trivial function that always returns one, for use with the constant
    shape model (S=1).

    Returns a list of two values: the worker number, and the number of flushes to the disk that have been preformed.
    This allows the master process to collate the data that has been saved by each worker node.
    """

    if isinstance(shape_func, bool):
        def shape_func(k1, k2, k3):
            return 1

    # shape_func = memoize(shape_func)

    # Read in the ell dataframe that has already been split up by the master worker
    ell_dataframe = pd.read_csv(str(folder) + '/ell_grid_' + str(worker_index) + '.csv')

    # Create a list for the results of the integration to be saved into
    result_list = []

    # Construct a list of unique ell values, which is used to build the splines for the transfer functions
    ell_list = np.unique(np.concatenate((ell_dataframe['ell1'], ell_dataframe['ell2'], ell_dataframe['ell3'])))

    # Create a blank dictionary that the memorized transfer functions splines will get stored in
    transfer_spline_list = {}

    # Creates local variable quad which points to SciPy quad, a Python speed performance increase trick
    quad = sciint.quad

    # Initiate the number of flushes that this worker has performed to the disk, as zero
    flush_counter = 0

    # Go through each ell and compute the spline for each, then using function memoization to help speed up evaluation
    for ell in ell_list:
        transfer_k, transfer_data = transfers[ell]  # boltzmann.get_transfer(ell)
        transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')
        transfer_spline_list[ell] = memoize(transfer_spline)

    # Since the specific transfer functions are no longer needed, manually delete them to help memory management
    del transfers, ell_list

    for index, row in enumerate(ell_dataframe.itertuples()):

        result, err = quad(bispectrum_integrand, 5.5, 10,
                           args=(row.ell1, row.ell2, row.ell3, transfer_spline_list[row.ell1],
                                 transfer_spline_list[row.ell2], transfer_spline_list[row.ell3], shape_func),
                           epsabs=1E-6, epsrel=1E-6, limit=5000)  # TODO: check error, add full_output option

        # Store the value of the integration in a dictionary, which then gets collated into a list
        temp = {'index': row.index, 'ell1': row.ell1, 'ell2': row.ell2, 'ell3': row.ell3, 'value': result, 'err': err}
        result_list.append(temp)

        # Periodically, save the integration values to the disk. Here this is done by default every 2500 integrations,
        # however this value can be easily changed for either many more flushes, or fewer.
        if (index + 1) % 2500 == 0:
            # Convert the current results list to a dataframe and then save it to the correct location
            result_list = pd.DataFrame(result_list)
            result_list.to_csv(str(folder) + '/output_flush_worker' + str(worker_index) + '_' + str(flush_counter) +
                               '.csv', index=False)
            # Reset the results list to empty once successfully flushed to disk
            result_list = []

            flush_counter += 1

    # Save the remaining output
    if len(result_list) > 0:
        result_list = pd.DataFrame(result_list)
        result_list.to_csv(str(folder) + '/output_flush_worker' + str(worker_index) + '_' + str(flush_counter) + '.csv',
                           index=False)
        flush_counter += 1

    # Manually delete several large variables, in the hope of trying to help memory management
    del ell_dataframe
    del result_list
    del transfer_spline_list

    return [worker_index, flush_counter]
