# Created by Alessandro Maraio on 12/03/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file allows for a direct comparison between my own CMB power spectrum calculation and
the one performed by CAMB when told the full inflationary power spectrum.
"""

import CosmologyClass as Cosmo
import BoltzmannClass as Boltz
import DatabaseClass as Db
import IntegrationClass as Int

import camb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.75, rc={'text.usetex': True})

# Load inflationary power spectrum data
twopf_k = np.logspace(-6, 1.7, num=401)
data = Db.Database('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadraticRepo/QuadRepo/output/Quadratic.twopf'
                   '-zeta/20200202T135205/data.sqlite', 'twopf')
twopf_data = data.get_dataframe()

# First get Cl values computed using CAMB
params = camb.CAMBparams()
params.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544)
params.set_for_lmax(2500, lens_potential_accuracy=0)

# Use CAMB's set_initial_power_table feature to get predictions of the Cl from the entire numerical power spectrum
# not just using the As ns simplification
params.set_initial_power_table(twopf_k, twopf_data['twopf'])

# Use CAMB to do the integration and extract values for the unlensed scalar Cls.
results = camb.get_results(params)
power_spec = results.get_cmb_power_spectra(params, CMB_unit='muK')
unlensed_Cells = power_spec['unlensed_scalar']
ells = np.arange(2, unlensed_Cells.shape[0])


# Now perform the calculation myself
cosmo = Cosmo.Cosmology(template='Planck 2018')
boltz = Boltz.BoltzmannCode('camb', cosmo)

twopf_int = Int.Integration(boltz, data)
ell, c_ell = twopf_int.integrate_power_spectrum()

# Plot the results of CAMB and my calculation
plt.figure(figsize=(13, 7))
plt.loglog(ells, unlensed_Cells[2:, 0], lw=2, color='g')  # CAMB plot
plt.loglog(ell, c_ell, lw=2, color='blue')  # My data
plt.title('CMB TT power spectrum for Quadratic model using CAMB and me')
plt.ylabel(r'$C_{\ell} \,\, \ell ( \ell + 1 ) / 2 \pi$')
plt.xlabel(r'$\ell$')
plt.tight_layout()


plt.figure(figsize=(13, 7))
plt.semilogx(ells[0:2499], unlensed_Cells[2:2501, 0]/c_ell[0:2499], color='b')
plt.title('Relative difference between me and CAMB')
plt.ylabel('Relative difference')
plt.xlabel(r'$\ell$')
plt.tight_layout()
plt.show() 
