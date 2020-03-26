# Created by Alessandro Maraio on 31/01/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is designed to implement the interface to a Boltzmann code
that is used to calculate the transfer functions of the temperature 
anisotropy of the CMB.

The goal here is to establish which code we are using (either CAMB
or CLASS) and then provide a consistent interface for the main code
as here we can deal with the specialisms of each code
"""


class BoltzmannCode:

    def __init__(self, code, cosmology):
        # Here we establish the BolzmannCode class for a given code
        # At the moment, we are only using CAMB - however a CLASS
        # implementation can be built later

        if code.lower() == 'camb':
            # Try to import camb first.
            try:
                import camb
            except ImportError:
                print('CAMB not found and  a mandatory requirement. \n' +
                      'Please install it (e.g. via pip or conda) before continuing.')
                from sys import exit
                exit()

            self.params = camb.CAMBparams(max_l=2500)  # (Want_CMB_lensing=False, DoLensing=False)
            self.params.set_cosmology(H0=cosmology.H0, ombh2=cosmology.omega_bh2,
                                      omch2=cosmology.omega_cdmh2, tau=cosmology.tau)

        elif code.lower() == 'class':
            raise RuntimeError('CLASS is currently not supported, please use CAMB instead.')

        else:
            raise RuntimeError('Unknown Boltzmann code specified.')

        self.transfer = None

    def __del__(self):
        # Manual class destructor that can be called at will
        del self.transfer
        del self.params

    def compute_transfer(self, accuracy_level=3, lSampleBoost=50):
        print('--- Computing transfer functions ---', flush=True)
        import camb
        self.params.set_accuracy(AccuracyBoost=accuracy_level, lSampleBoost=lSampleBoost)
        self.transfer = camb.get_transfer_functions(self.params).get_cmb_transfer_data()
        print('--- Computed transfer functions ---', flush=True)

    def get_transfer(self, ell, **kwargs):
        if self.transfer is None:
            self.compute_transfer(**kwargs)

        ell_convert = {}

        for index, ell_value in enumerate(self.get_ell_list()):
            ell_convert[ell_value] = index

        if ell in ell_convert:
            return self.transfer.q, self.transfer.delta_p_l_k[0, ell_convert[ell], :]
        else:
            raise RuntimeError('CAMB did not compute the transfer function for the required ell value of ' + str(ell) +
                               '\nPlease re-run CAMB with a higher lSampleBoost value to compute the required ell value')

    def get_ell_list(self, **kwargs):
        if self.transfer is None:
            self.compute_transfer(**kwargs)

        return self.transfer.L
