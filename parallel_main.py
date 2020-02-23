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
from mayavi import mlab
import plotly.graph_objects as go


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
        transfer_k, transfer_data = boltz.get_transfer(ell-2)
        transfer_list[ell] = [transfer_k, transfer_data]

    threepf_int = Bispec.Bispectrum(boltz, data)

    # Build a grid of allowed values for comparison with Shellard CMB paper
    grid = Bispec.build_grid(ell_sum=4000, ell_cut=1950, ell_step=5)

    # Split grid up into chuncks that are sent to each process
    split_grid = np.array_split(grid, size-1)

    # Build a MPI Pool instance to assign work to
    executor = MPIPoolExecutor(max_workers=size+1)
    sys.stdout.flush()

    start_time = time.time()

    # Distribute the integration tasks over the available workers. Note: we don't care about the return
    # order, as pandas and matplotlib can deal with un-ordered data.
    result = executor.map(Bispec.parallel_integrate, split_grid, itertools.repeat(transfer_list), unordered=True)

    # Turn list of results into a pandas dataframe
    temp = []
    for i in result:
        temp += i

    data = pd.DataFrame(temp)

    # Print integration summary statistics
    print('--- Finished bispectrum integration ---')
    finish_time = time.time()
    print('--- Bispectrum took ' + str(finish_time - start_time) + ' seconds ---')
    print('--- with an average of ' + str(len(temp) / (finish_time - start_time)) + ' samples / second ---')
    sys.stdout.flush()

    # Begin visualisations of data

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ColourMap = plt.cm.viridis

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(data['ell1'], data['ell2'] - data['ell3'], data['value'], cmap=ColourMap, antialiased=False)
    ax.set_xlabel('ell 1')
    ax.set_ylabel('ell 2 - ell 3')
    ax.set_zlabel('Value', labelpad=15)
    plt.show()
