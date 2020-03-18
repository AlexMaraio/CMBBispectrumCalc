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
import pandas as pd
import itertools

import CosmologyClass as Cosmo
import BoltzmannClass as Boltz
import DatabaseClass as Db
import Bispectrum as Bispec


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # gives number of cores available
    rank = comm.Get_rank()

    print('Number of available processes: ' + str(size), flush=True)

    # Get database and compute transfer functions from standard cosmology
    data = Db.Database('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadraticRepo/QuadRepo/output/Quadratic.twopf-zeta/20200202T135205/data.sqlite',
        'twopf')

    cosmo = Cosmo.Cosmology(template='Planck 2015')

    boltz = Boltz.BoltzmannCode('camb', cosmo)

    boltz.compute_transfer()

    # Store the transfer function data as a dictionary indexed by the ell value.
    transfer_list = {}
    for ell in boltz.get_ell_list():
        transfer_k, transfer_data = boltz.get_transfer(ell)
        transfer_list[ell] = [transfer_k, transfer_data]

    threepf_int = Bispec.Bispectrum(boltz, data)

    const_ell_sum_grid = False
    ell_isosurface_grid = True

    integration_type = 'const_ell_sum_grid' if const_ell_sum_grid else 'ell_isosurface_grid'

    if const_ell_sum_grid:
        # Build a grid of allowed values where ell1+ell2+ell3 is a constant for comparison with Shellard CMB paper
        grid = Bispec.build_grid_ell_sum(ell_sum=4000, ell_cut=1950, ell_step=10)

    elif ell_isosurface_grid:
        # Build a more relaxed grid, in which the ell values only need to pass the triangle conditions
        grid = Bispec.build_ell_grid(ell_max=2000, ell_step=25)

    else:
        raise RuntimeError('No grid type specified to build, so can not do a bispectrum integration!')

    # Split grid up into chuncks that are sent to each process
    split_grid = np.array_split(grid, size-1)

    # Get current timestamp and save it in a user friendly way.
    # Then using this timestamp and the selected integration type, build up a string 'folder' which is where
    # input/output dats is saved from the integration
    t = time.localtime()
    timestamp = time.strftime('%Y%m%dT%H%M%S', t)
    folder = str(integration_type) + '_' + str(timestamp)

    # Makes the folder to save the data in
    os.mkdir(folder)

    # Save each chunck of data to a csv file, which can then be read in by each workder
    for index, split in enumerate(split_grid):
        split.to_csv(str(folder) + '/ell_grid_' + str(index) + '.csv', index=False)

    # Manually remove grid and split_grid now that they aren't needed, trying to manually help memory management
    del grid, split_grid

    # Build a MPI Pool instance to assign work to
    executor = MPIPoolExecutor(max_workers=size+1)
    sys.stdout.flush()

    print('--- Starting bispectrum integration ---', flush=True)
    start_time = time.time()

    # Distribute the integration tasks over the available workers. Note: we don't care about the return
    # order, as pandas and matplotlib can deal with un-ordered data.
    # result = executor.map(Bispec.parallel_integrate, split_grid, itertools.repeat(transfer_list), unordered=True)

    result = executor.map(Bispec.parallel_integrate, np.arange(size - 1), itertools.repeat(folder),
                          itertools.repeat(transfer_list), unordered=True)

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
