# Created by Alessandro Maraio on 06/04/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file  is dedicated to the CMB map generation

TODO: something about folders, or saving the data somewhere in particular?
"""

import os
import time
import ctypes
import pandas as pd
import numpy as np
from scipy import interpolate as interp
import camb
import matplotlib.pyplot as plt
import seaborn as sns
import healpy

from lib import DatabaseClass as Db

# Enable seaborn plot styling in figures
sns.set(font_scale=1.75, rc={'text.usetex': True})


class CMBMap:

    def __init__(self, lib_dir):
        """
        Class constructor.

        Args:
            lib_dir (str): The location of the libCppTollKit.so shared library that contains the functions implemented
                           in C++ for use with map making
        """

        self.inflation_power_spec = None
        self.bispectrum = None
        self.alm = None
        self.alm_df = None

        if not os.path.isfile(os.path.abspath(lib_dir)):
            raise RuntimeError('The specified path to the libCppToolKit.so shared library can not be found. '
                               'Please compile the necessary C++ & Fortran library and then re-run.')

        self.library = ctypes.CDLL(os.path.abspath(lib_dir))

        # Set the argument types of the calc_alm2 function in the C++ code
        self.library.calc_alm2.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float)]

        # Also set its return type to a void.
        self.library.calc_alm2.restype = ctypes.c_void_p

    def set_cmb_bispectrum(self, bispectrum):
        """
        This function is designed to set the CMB bispectrum to one that has been previously calculated

        Args:
            bispectrum (str): Path to the bispectrum

        Returns:
            None
        """

        # Check that the path exists
        if not os.path.isfile(os.path.abspath(bispectrum)):
            raise RuntimeError('The provided bispectrum file does not exist! Please provide the correct '
                               'file-name and re-run.')

        # Save the path in the class
        self.bispectrum = os.path.abspath(bispectrum)

    def set_inflation_power_spec_db(self, power_spec_db):
        """
        Function to set the inflationary power spectrum class

        Args:
            power_spec_db: A Database class instance that holds the inflationary twopf database and k table

        Returns:
            None
        """

        self.inflation_power_spec = power_spec_db

    def generate_ell_interp_grid(self):
        """
        Generates the ell grid that is going to be interpolated over. Uses C++ to make the loops much faster than
        the Python version.
        """

        # Use the ctypes library to set the argument types for the build_ell_grid_cpp function.
        self.library.build_ell_grid_cpp.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_bool]

        # Use the ctypes library to transform a Python string into a C char pointer, so that way it can be read in
        file_path = ctypes.c_char_p(b'interpolated_grid.csv')

        # Call the function in the C++ library, which outputs the grid to the desired location above.
        # Note: arguments are: ell_max, ell_step, file_path, full_grid
        self.library.build_ell_grid_cpp(500, 1, file_path, False)

    def interpolate_bispectrum(self):
        """
        Function to interpolate the given bispectrum over the provided full grid of configurations.
        """

        print('--- Reading in existing CMB bispectrum ---')

        # Read in the existing CMB bispectrum data.
        data = pd.read_csv(self.bispectrum)

        # Transform the data into units of mu K^3, and remove the Fergusson & Shellard normalisation to get raw values
        data['value'] = data['value'] * ((2.725 * 1E6) ** 3)
        data['value'] = data['value'] / ((2 * data.ell1 + 1) * (2 * data.ell2 + 1) * (2 * data.ell3 + 1) *
                                         (data.ell1 + data.ell2 + data.ell3))

        # Since we are using the SciPy griddata function, we have to turn our 1D list of values into a different
        # form of list, so that way it can interpolate correctly.

        # First, create two temporary lists to hold the set of points and values in
        points = []
        values = []

        # Now iterate through the data and store the data in the lists
        for iter_row in data.itertuples():
            temp = [iter_row.ell1, iter_row.ell2, iter_row.ell3]
            points.append(temp)
            values.append(iter_row.value)

        # Now that we've saved the data, we can remove the raw data from memory
        del data

        # Transform values into a NumPy array, for interpolation
        values = np.array(values)

        print('--- Reading in the interpolated grid ---')

        # Read in the grid of ells that we're going to interpolate over
        ell_grid = pd.read_csv('interpolated_grid.csv')

        print('--- Performing interpolation on grid ---')

        # Call the grid data function to interpolate over.
        ell_grid['data'] = interp.griddata(points, values, (ell_grid['ell1'], ell_grid['ell2'], ell_grid['ell3']),
                                           fill_value=0)

        print('--- Saving interpolation output ---')

        # Save the data again, but now with the interpolated values in place.
        ell_grid.to_csv('interpolated_grid.csv', index=False)

        # Remove all unnecessary variables now that the data has been saved.
        del ell_grid, points, values

    def generate_cl_alm(self):
        # First check that the inflationary power spectrum has been provided
        if self.inflation_power_spec is None:
            raise RuntimeError('Please first specify the inflationary twopf database for which you want to '
                               'produce a map for.')

        print('--- Generating cl values ---')

        # Now, use CAMB to generate the cl values for the specified model
        params = camb.CAMBparams()
        params.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544)
        params.set_for_lmax(2500)
        params.set_initial_power_table(self.inflation_power_spec.get_k_table_db['k_physical'],
                                       self.inflation_power_spec.get_dataframe['twopf'])

        # Perform CAMB computation
        results = camb.get_results(params)
        power_spec = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True)

        # Extract the cl values from the results
        cl = power_spec['unlensed_scalar'][:, 0]

        # Produce first map using just cl's
        map1 = healpy.sphtfunc.synfast(cl, 256)
        healpy.visufunc.mollview(map1, cmap='jet', title='Gaussian Map', unit=r'$\mu$K')
        plt.show(block=False)

        # Now turn the cl values into alm's
        alm = healpy.sphtfunc.synalm(cl, new=True)
        self.alm = alm

        # Produce map using alm's, for cross-check
        map2 = healpy.sphtfunc.alm2map(alm, 256)
        healpy.visufunc.mollview(map2, cmap='jet', title='Gaussian Map', unit=r'$\mu$K')
        plt.show(block=False)

        print('--- Saving cl and alm\'s ---')

        # Save cl's to csv - needed to be read in by  the C++ code
        cl_df = pd.DataFrame({'ell': range(2500), 'cl': cl[0:2500]})  # TODO: make this ell_max adjustable
        cl_df.to_csv('cl.csv', index=False)

        # Then save the alm's to a csv too
        alm_list = []
        for i in range(len(alm)):
            ell, m = healpy.sphtfunc.Alm.getlm(lmax=2550, i=i)
            alm_list.append({'ell': ell, 'm': m, 'alm_re': np.real(alm[i]), 'alm_im': np.imag(alm[i])})

        alm = pd.DataFrame(alm_list)
        alm.to_csv('alm.csv', index=False)

        # Store the alm dataframe in the class
        self.alm_df = alm


if __name__ == '__main__':
    """
    This main function uses the CMBMap class to produce a non-Gaussian CMB map for the specified model.
    """

    # Initialise the CMBMap class, and point it to the libCppToolKit shared library
    cmb = CMBMap(lib_dir='./cpp/build/libCppToolKit.so')

    # Set the CMB bispectrum, to a previously calculated output file
    cmb.set_cmb_bispectrum('/home/amaraio/Documents/CMBBispectrumCalc/OldData/'
                           'bispectrum_ellmax_745_2020-03-31_022214.csv')

    # Generate the ell grid to interpolate over
    cmb.generate_ell_interp_grid()

    # Interpolate the bispectrum on the provided grid
    cmb.interpolate_bispectrum()

    # Create a twopf database class, that points to an existing inflationary power spectrum database
    twopf_data = Db.Database('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadCMB/TEST2/output/thingy_zeta_twopf/'
                             '20200325T150327/data.sqlite', 'twopf')

    # Set the k_table for the above data
    twopf_data.set_k_table('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadCMB/TEST2/k_table.dat')

    # Set the CMB map class to use this inflationary power spectrum for the map making
    cmb.set_inflation_power_spec_db(twopf_data)

    # Generate and save the cl & alm values, for the provided inflationary power spectrum
    cmb.generate_cl_alm()

    # Call the initialise function in the C++ library, which stores the cl and alm values in it
    cmb.library.initalise()

    # Use the ctypes library so that way we can pass floats to and from the C++ code
    alm_re = ctypes.c_float()
    alm_im = ctypes.c_float()

    # alms = pd.read_csv('alm.csv')

    # Record the start time, so that way we can log how long the calculation takes
    start_time = time.time()

    print('--- Starting computation of non-Gaussian alm\'s ---')

    # Create a blank array of zeros so that way we can store the non-Gaussian alm corrections in separately
    alm_zero = np.zeros(np.size(cmb.alm), dtype=complex)

    # Customisable ell_max value that sets the level of precision in the calculation.
    # Note: the calculation starts to take a really long time when this is about > 250
    ell_max = 100

    # Iterate through the alm list
    for index, row in enumerate(cmb.alm_df.itertuples()):
        # Check that the ell is less than our critical ell, otherwise ignore its effects
        if row.ell > ell_max:
            continue

        # Use the shared library to compute the bispectrum corrections
        cmb.library.calc_alm2(row.ell, row.m, ell_max, ctypes.byref(alm_re), ctypes.byref(alm_im))

        # Adjust the original alm parameters accordingly
        cmb.alm[index] += 50000 * (alm_re.value + alm_im.value * 1j)
        # idx = healpy.sphtfunc.Alm.getidx(lmax=2500, l=row.ell, m=row.m)
        alm_zero[index] = 50000 * (alm_re.value + alm_im.value * 1j)

    # Print summary statistics
    print('--- Finished computation ---')
    finish_time = time.time()
    print('--- Computation took ' + str(round(finish_time - start_time, 2)) + ' seconds ---')

    # Now use healpy to generate a new map, with non-Gaussian corrections
    alms = np.array(cmb.alm)
    map_arr1 = healpy.sphtfunc.alm2map(alms, 256)
    healpy.visufunc.mollview(map_arr1, cmap='jet', title='Non-Gaussian corrections', unit=r'$\mu$K')
    plt.show(block=False)

    # Plot just the non-Gaussian corrections
    alms = np.array(alm_zero)
    map_arr2 = healpy.sphtfunc.alm2map(alms, 256)
    healpy.visufunc.mollview(map_arr2, cmap='jet', title='Just non-Gaussian corrections', unit=r'$\mu$K')
    plt.show()
