# Created by Alessandro Maraio on 23/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is the one that should be run if we want to use the parallel version of the CMB bispectrum integration.
This implementation uses MPI via Mpi4Py, and so should scale freely with available cores, which hopefully
means that HPC integration should be simple.

It should be run with the following command:
mpiexec -n numprocs python -m mpi4py.futures parallel_main.py
"""


import sys
import os
import time
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np
from lib.Interpolation import RadialBasisFunction as Rbf
import pandas as pd
import itertools

from lib import BoltzmannClass as Boltz, DatabaseClass as Db, CosmologyClass as Cosmo, Grid, Bispectrum as Bispec

if __name__ == '__main__':
    # Initiate MPI4Py variables needed for the parallel computation
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # Gives number of cores available
    rank = comm.Get_rank()  # Get rank that this processor is currently

    # Print number of processors available to ensure that MPI is working correctly
    print('Number of available processes: ' + str(size), flush=True)

    # Booleans to set which type of integration we want to perform
    # NOTE: only one of them is meant to be True at any one time
    const_ell_sum_grid = False
    ell_volume_grid = True

    if const_ell_sum_grid == ell_volume_grid:
        raise RuntimeError('Both types of integration can not be specified at the same time. Please run with only '
                           'one type selected and then re-run with the other one afterwards.')

    # Switch to use the inflationary bispectrum in the integration or not.
    # If False, the uses the constant shape model where is is simply set to unity: S=1
    use_inflationary_bispectrum = False

    # Creates a string which is derived from the integration type switches
    integration_type = 'ell_sum_grid' if const_ell_sum_grid else 'ell_volume_grid'

    # Get current timestamp and save it in a user friendly way.
    # Then using this timestamp and the selected integration type, build up a string 'save_folder' which is where
    # input/output data is saved from the integration
    t = time.localtime()
    timestamp = time.strftime('%Y%m%dT%H%M%S', t)
    folder = 'OutputData/' + str(integration_type) + '_' + str(timestamp)

    if not os.path.isdir('OutputData'):
        os.mkdir('OutputData')

    # Makes the save_folder to save the data in
    os.mkdir(folder)

    # Do we want to use the inflationary bispectrum in our CMB bispectrum integration?
    # If not, we use the constant shape model, where S=1 on all configurations
    if use_inflationary_bispectrum:
        # Set up zeta three-point function database, which is an output of a CppTransport CMB task
        data = Db.Database('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadCMB/TEST_SMALLER/output/'
                           'thingy_threepf_zeta/20200325T204036/data.sqlite', 'threepf')

        # Provide the path needed for the k_table to be read in
        data.set_k_table('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadCMB/TEST_SMALLER/k_table.dat')

        # Save the inflationary bispectrum into the correct save_folder after using splines on the k values
        data.save_inflation_bispec(folder)

        # Read in the saved inflationary bispectrum to a pandas dataframe
        inf_bispec = pd.read_csv(str(folder) + '/inflationary_bispectrum_data.csv')

        # Transform k values into log space, which makes the interpolation much nicer
        inf_bispec['k1'] = np.log10(inf_bispec['k1'])
        inf_bispec['k2'] = np.log10(inf_bispec['k2'])
        inf_bispec['k3'] = np.log10(inf_bispec['k3'])

        # Also divide by the minimum threepf value to normalise values nicely
        threepf_min = np.min(inf_bispec['threepf'])
        inf_bispec['threepf'] = inf_bispec['threepf'] / threepf_min

        print('--- Starting interpolation of inflationary bispectrum ---', flush=True)

        # Use SciPy radial basis function interpolation to build interpolate over the k1, k2, k3 grid
        # Here, we are using the "function='quintic'" option, which has previously been found to work
        # much better than other functions using the same Rbf
        # shape_func = interp.Rbf(inf_bispec['k1'], inf_bispec['k2'], inf_bispec['k3'], inf_bispec['threepf'],
        #                         function='quintic')

        shape_func = Rbf.Rbf(inf_bispec['k1'], inf_bispec['k2'], inf_bispec['k3'], inf_bispec['threepf'],
                             function='quintic')

        print('--- Inflationary bispectrum interpolated ---', flush=True)

        # Now that the inflationary bispectrum is saved, it is not needed any more.
        # Manually deleting variables to try and help memory management
        del data, inf_bispec

    else:
        # Here, we do not wish to use the provided inflationary bispectrum, so use constant shape model instead
        shape_func = False

        # No need to normalise the inflationary bispectrum if not using it
        threepf_min = 1

    # Establish a grid class that will be used to construct the desired (ell1, ell2, ell3) grid for integraiton
    grid = Grid.Grid()

    # Use the grid class to build up an ell volume up to ell max in ell steps.
    # Also available is build_ell_sum_grid
    grid.build_ell_volume_grid(ell_step=15, ell_max=500)

    # Initiate string which is where we will read & write transfer function data to
    transfer_folder = 'transfers'

    # Try and find the number of files in the folder where the transfers are saved into, if this does not exist
    # then set the number of items to zero
    try:
        num_transfers = len(os.listdir(str(transfer_folder))) + 1

    except FileNotFoundError:
        num_transfers = 0

    # If we have an ell max that is higher than the number of transfers saved, then we will have to re-compute them
    if grid.ell_max > num_transfers:
        # Set up cosmology class from Planck 2018 template
        cosmo = Cosmo.Cosmology(template='Planck 2018')

        # Create Boltzmann class using CAMB as the solver, using the above cosmological values
        boltz = Boltz.BoltzmannCode('camb', cosmo)

        # Compute the transfer functions using the provided Boltzmann code
        boltz.compute_transfer()

        # Save the transfer functions to the disk using the .npy format, which saves time re-computing them later
        boltz.save_transfers(transfer_folder)

        # Store the transfer function data as a dictionary indexed by the ell value.
        transfer_list = {}
        for ell in boltz.get_ell_list():
            transfer_k, transfer_data = boltz.get_transfer(ell)
            transfer_list[ell] = [transfer_k, transfer_data]

        # Now that transfers are saved in a dictionary, the Boltzmann class is not needed any more.
        # Manually delete the class to try and save RAM
        del boltz

    # Else, we can simply read in the previously calculated transfer function data
    else:
        print('--- Reading in transfer functions from disk ---', flush=True)
        transfer_list = Boltz.read_transfers(transfer_folder)

    # Flush any output stream to the terminal
    sys.stdout.flush()

    # Randomly shuffle the data grid, which should evenly distribute the easier and harder integrations between cores
    grid = grid.get_grid.sample(frac=1).reset_index(drop=True)

    # Split grid up into chuncks that are sent to each process
    split_grid = np.array_split(grid, size - 1)

    # Save each chunck of data to a csv file, which can then be read in by each worker
    for index, split in enumerate(split_grid):
        split.to_csv(str(folder) + '/ell_grid_' + str(index) + '.csv', index=False)

    # Manually remove grid and split_grid now that they aren't needed, trying to manually help memory management
    del grid, split_grid

    # Build a MPI Pool instance to assign work to
    executor = MPIPoolExecutor(max_workers=size+1)
    sys.stdout.flush()

    print('--- Starting bispectrum integration ---', flush=True)
    start_time = time.time()

    # Distribute the integration tasks over the available workers.
    # Note: we don't care about the return order, as pandas and matplotlib can deal with un-ordered data.
    # result = executor.map(Bispec.parallel_integrate, split_grid, itertools.repeat(transfer_list), unordered=True)

    result = executor.map(Bispec.parallel_integrate, np.arange(size - 1), itertools.repeat(folder),
                          itertools.repeat(transfer_list), itertools.repeat(shape_func),
                          unordered=True)

    # Turn list of results into a pandas dataframe
    data = []
    for i in result:
        data.append(i)

    worker_df_list = []

    # New way of combining output data
    for output in data:
        worker = output[0]
        flushes = output[1]

        temp_df = []

        # Go through each flush from each worker read that in as a pandas dataframe
        for flush in np.arange(flushes):
            tempdf = pd.read_csv(str(folder) + '/output_flush_worker' + str(worker) + '_' + str(flush) + '.csv')
            temp_df.append(tempdf)
            # Once each file has been added to the list, it is no longer needed - and so can be deleted
            os.remove(str(folder) + '/output_flush_worker' + str(worker) + '_' + str(flush) + '.csv')

        # Join the dataframes together to get one for each worker
        temp_df = pd.concat(temp_df, ignore_index=True)

        worker_df_list.append(temp_df)

    # Then combine each dataframe from each worker into one that has all the data in
    data = pd.concat(worker_df_list, ignore_index=True)

    # Remove normalisation to give the correct values after integration
    data['value'] = data['value'] * threepf_min
    data['err'] = data['err'] * threepf_min

    # TODO: raise a warning if the value/error ratio is greater than, say, 10 as integration may be unreliable
    # and need to tighten the error tolerances

    # Manually delete temporary lists that are no longer needed, trying to manually help memory management
    del worker_df_list, temp_df

    # Print integration summary statistics
    print('--- Finished bispectrum integration ---')
    finish_time = time.time()
    print('--- Bispectrum took ' + str(round(finish_time - start_time, 2)) + ' seconds ---')
    print('--- with an average of ' + str(round(data.shape[0] / (finish_time - start_time), 2)) +
          ' samples / second ---')
    sys.stdout.flush()

    # Begin visualisations of data

    if const_ell_sum_grid:
        # Visualise the data for the constant ell sum grid
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Colour map that will be used to make plots. Freely adjustable to any Matplotlib map from their library
        ColourMap = plt.cm.viridis

        # Get the constant ell value that this data grid corresponds to, for plot title and file name uses
        const_ell = data['ell1'].iloc[0] + data['ell2'].iloc[0] + data['ell3'].iloc[0]

        # Get current user timestamp to label data with
        t = time.localtime()
        timestamp = time.strftime('%Y-%m-%d_%H%M%S', t)

        # Save data to csv format. Both for backup and reading into plotting tools
        data.to_csv(str(folder) + '/bispectrum_contell_' + str(const_ell) + '_' + str(timestamp) + '.csv', index=False)

        # Plot the data on a grid with x=ell1 and y=ell2 - ell3, which allows for somewhat easy viewing of the data
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        surf = ax.plot_trisurf(data['ell1'], data['ell2'] - data['ell3'], data['value'], cmap=ColourMap, antialiased=False)
        ax.set_xlabel('ell 1')
        ax.set_ylabel('ell 2 - ell 3')
        ax.set_zlabel('Value', labelpad=15)
        ax.set_title('Bispectrum plotted for constant ell1 + ell2 + ell3 = ' + str(const_ell))

        cbar = fig.colorbar(surf, ax=ax, shrink=1, aspect=17)
        cbar.ax.get_yaxis().labelpad = 5
        cbar.ax.set_ylabel('value', rotation=90)

        plt.tight_layout()
        plt.savefig(str(folder) + '/constant_ell_' + str(const_ell) + '.png', dpi=500)
        plt.show()

    elif ell_isosurface_grid:
        # Visualise the data for the generic ell1, ell2, ell3 grid
        # TODO: find a better way to visualise the isosurfaces as the current matplotlib implementation is lacking
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.colors as colors

        ColourMap = 'seismic'

        # Get the ell max value that was used to build the grid, for things like plots and saving data
        ell_max = np.max(data['ell1'])

        # Get current user timestamp to label data with
        t = time.localtime()
        timestamp = time.strftime('%Y-%m-%d_%H%M%S', t)

        # Save data to csv format. Both for backup and reading into plotting tools
        data.to_csv(str(folder) + '/bispectrum_ellmax_' + str(ell_max) + '_' + str(timestamp) + '.csv', index=False)

        # First plot, with a symmetric log-normal color map setting, which brings out more detail
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(data['ell1'], data['ell2'], data['ell3'], c=data['value'], cmap=ColourMap,
                         norm=colors.SymLogNorm(linthresh=0.0075, linscale=1, vmin=-0.0275, vmax=0.0275))

        cbar = fig.colorbar(img, shrink=1, aspect=17)
        cbar.ax.get_yaxis().labelpad = 5
        cbar.ax.set_ylabel('value', rotation=90)

        plt.tight_layout()
        plt.savefig(str(folder) + '/Isosurface_ellmax_' + str(ell_max) + '_1.png', dpi=500)
        plt.show(block=False)

        # Second plot, with normal colour map settings, which is better for extreme values
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(data['ell1'], data['ell2'], data['ell3'], c=data['value'], cmap=ColourMap)

        cbar = fig.colorbar(img, shrink=1, aspect=17)
        cbar.ax.get_yaxis().labelpad = 5
        cbar.ax.set_ylabel('value', rotation=90)

        plt.tight_layout()
        plt.savefig(str(folder) + '/Isosurface_ellmax_' + str(ell_max) + '_2.png', dpi=500)
        plt.show()
