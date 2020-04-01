# Created by Alessandro Maraio on 31/01/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is the main file where we demonstrate some of the main features in my code that can be run from a single
core, ie. using the normal Python command line command

python main.py

Here, we will first perform a CMB power spectrum integral to obtain the Cl values, and then go on to perform
a bispectrum integral using equal ell's.
"""


from lib import BoltzmannClass as Boltz, DatabaseClass as Db, CosmologyClass as Cosmo, IntegrationClass as Int, \
    Bispectrum as Bispec, Visualisation as Viz
import os
import time


# First, we will compute the CMB power spectrum given an inflationary power spectrum computed using CppTransport

# Import the twopf data that was generated from a CppTransport CMB task
data = Db.Database('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadCMB/TEST2/output/thingy_zeta_twopf/'
                   '20200325T150327/data.sqlite', 'twopf')

# Set the k_table for the above data
data.set_k_table('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadCMB/TEST2/k_table.dat')

# Set the cosmology template for this example
cosmo = Cosmo.Cosmology(template='Planck 2018')

# Set up Boltzmann solver and compute transfer functions
boltz = Boltz.BoltzmannCode('camb', cosmo)
boltz.compute_transfer()

# Execute the power spectrum integration
twopf_int = Int.Integration(boltz, data)
ell, c_ell = twopf_int.integrate_power_spectrum()

# Plot the power spectrum using seaborn with LaTeX labels
Viz.twopf_visualisation(ell, c_ell, use_LaTeX=True)


# We will now perform a CMB bispectrum integral, only using equal ells though (where ell1=ell2=ell3),
# and in the constant shape model where S=1

# Create bispectrum class with the boltzmann class and twopf data
threepf_int = Bispec.Bispectrum(boltz, data)

# Perform the equal ell integration
results = threepf_int.integrate_constant_ell(ell_step=2, ell_max=2500)

# Get current timestamp and save it in a user friendly way.
# Then using this timestamp, build up a string 'save_folder' which is where output data is saved from the integration
t = time.localtime()
timestamp = time.strftime('%Y%m%dT%H%M%S', t)
folder = 'equal_ell_grid_step' + str(threepf_int.ell_step) + '_' + str(timestamp)

# Makes the save_folder to save the data in
os.mkdir(folder)

# Save the integration data into the folder
results.to_csv(str(folder) + '/bispectrum_equalell_step_' + str(threepf_int.ell_step) + '_' + str(timestamp) + '.csv',
               index=False)

# Plot it, and save to the folder
Viz.equal_ell_bispectrum_plot(results['ell'], results['value'], use_LaTeX=True, save_folder=folder)


print('Length of ell array: ' + str(len(ell)))

Viz.twopf_visualisation(ell, c_ell, useLaTeX=True)

