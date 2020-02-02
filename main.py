# Created by Alessandro Maraio on 31/01/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


import CosmologyClass as Cosmo
import BoltzmannClass as Boltz
import DatabaseClass as Db
import IntegrationClass as Int
import Visualisation as Viz


# data = Db.Database('/home/amaraio/Documents/CMBBispectrumCalc_Repos/DoubleQuadRepo/DQuadRepo_tk2_2/output/dquad.twopf-zeta/20200202T124829/data.sqlite', 'twopf')
data = Db.Database('/home/amaraio/Documents/CMBBispectrumCalc_Repos/QuadraticRepo/QuadRepo/output/Quadratic.twopf-zeta/20200202T135205/data.sqlite', 'twopf')


cosmo = Cosmo.Cosmology(template='Planck 2018')

boltz = Boltz.BoltzmannCode('camb', cosmo)

twopf_int = Int.Integration(boltz, data)
ell, c_ell = twopf_int.integrate_power_spectrum()

print('Length of ell array: ' + str(len(ell)))

Viz.twopf_visualisation(ell, c_ell, useLaTeX=True)

