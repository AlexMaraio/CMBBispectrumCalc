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


import os
import glob
import numpy as np


class BoltzmannCode:

    def __init__(self, code, cosmology):
        """
        Constructor for the BoltzmannCode class.
        This is designed to provide a single interface for the code to interact with the Boltzmann codes.
        At the moment, only a CAMB interface is implemented, however this can be easily extended to use CLASS in the
        future, if desired.

        Args:
            code (str): The Boltzmann code to use. Only CAMB is supported (see above).
            cosmology (Cosmology class): A cosmology class that stores that values of certain cosmological parameters
                in.
        """

        if code.lower() == 'camb':
            # Try to import CAMB to see if it is installed.
            try:
                import camb
            except ImportError:
                print('CAMB not found and  a mandatory requirement. \n' +
                      'Please install it (e.g. via pip or conda) before continuing.')
                from sys import exit
                exit()

            # Now that we are certain that CAMB is installed, we can start configuring it with the
            # specified cosmology values from the cosmology class
            self.params = camb.CAMBparams(max_l=2500)
            self.params.set_cosmology(H0=cosmology.H0, ombh2=cosmology.omega_bh2,
                                      omch2=cosmology.omega_cdmh2, tau=cosmology.tau)

        elif code.lower() == 'class':
            raise RuntimeError('CLASS is currently not supported, please use CAMB instead.')

        else:
            raise RuntimeError('Unknown Boltzmann code specified.')

        # Initialise the transfer variables, which is where transfer function data will be stored into
        self.transfer = None

    def __del__(self):
        # Manual class destructor that can be called at will
        del self.transfer
        del self.params

    def compute_transfer(self, accuracy_level=3, lSampleBoost=50):
        """
        Function to compute the transfer functions from the Boltzmann code

        Args:
            accuracy_level (int): The overall accuracy level for computing the transfer functions, as used by
                the CAMB code. Anything in the range 3-5 should provide sufficient accuracy for the bispectrum.
            lSampleBoost (int): Parameter that determines how many transfer functions to compute. By setting this
                to 50, we force CAMB to compute all the transfer functions up to ell=2500
        """

        print('--- Computing transfer functions ---', flush=True)

        # Ensure that CAMB is imported
        import camb

        # Set the CAMB accuracy parameters to the ones given
        self.params.set_accuracy(AccuracyBoost=accuracy_level, lSampleBoost=lSampleBoost)

        # Compute the transfer functions, and then save them in the class
        self.transfer = camb.get_transfer_functions(self.params).get_cmb_transfer_data()

        print('--- Computed transfer functions ---', flush=True)

    def get_transfer(self, ell, **kwargs):
        """
        Function that returns the transfer function for the given ell value.

        Args:
            ell (int): l value of the transfer function that is returned.
            **kwargs: Additional parameters passes to compute_transfers, if not already computed

        Returns:
            Two lists:
                - The first list is the k values that the transfer function is evaluated at
                - The second is the value of the transfer function at these points
        """

        # Ensure that the transfer functions have already been computed, as if not then we compute them
        if self.transfer is None:
            self.compute_transfer(**kwargs)

        # Create an empty dictionary that allows us to relate CAMB index values to actual ell values
        ell_dict = {}

        for index, ell_value in enumerate(self.get_ell_list()):
            ell_dict[ell_value] = index

        if ell in ell_dict:
            return self.transfer.q, self.transfer.delta_p_l_k[0, ell_dict[ell], :]
        else:
            raise RuntimeError('CAMB did not compute the transfer function for the required ell value of ' + str(ell) +
                               '\nPlease re-run CAMB with a higher lSampleBoost value to compute the required ell value')

    def get_ell_list(self, **kwargs):
        """
        Function for returning the CAMB ell list of the transfer functions that it has computed

        Args:
            **kwargs: Key-word arguments passed to compute_transfer, if not already computed

        Returns:
            List of ell values from CAMB
        """

        if self.transfer is None:
            self.compute_transfer(**kwargs)

        return self.transfer.L

    def save_transfers(self, folder='transfers'):
        """
        Function that saves the transfer functions data to the folder 'transfers' in the current working directory.
        Each transfer function is saved to a .npy file, which is the proprietary NumPy format for saving arrays as,
        which gives quick and easy saving/loading.
        The transfer functions are saved with each ell that they correspond to.

        Args:
            folder (str): Optional argument which is the folder that the data will get saved into. Useful for caching
                          transfer functions for different cosmologies etc

        Returns:
            None: Does not return anything
        """

        # If the transfer functions are already not computed, then compute them now
        if self.transfer is None:
            self.compute_transfer()

        # If the sub-folder ./transfers does not exist, create it
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Go through each ell in the transfer functions and save them individually
        for index, ell in enumerate(self.get_ell_list()):
            transfer_k, transfer_data = self.get_transfer(ell)

            # We do not need to save the k-range for each transfer - saving it once at the start is fine
            if not os.path.isfile(str(folder) + '/transfer_k.npy'):
                np.save(str(folder) + '/k_data', transfer_k)

            # Save the transfer functions as .npy files
            np.save(str(folder) + '/transfer_' + str(ell), transfer_data)


def read_transfers(folder='transfers'):
    """
    File to read in the transfer functions from a folder.

    Args:
        folder (str): Optional argument which is where the transfer functions will be read in from

    Returns:
         transfer_dict (dict): A dictionary indexed by the ell values that has the k_data and transfer_data in each
                               entry
    """

    # Checks if the provided folder exists, and if it does not - then raise an error
    if not os.path.exists(folder):
        raise RuntimeError('The transfer functions have not been saved yet! Please re-run the dedicated '
                           'save-transfer script in order to use PyPy.')

    # Get the list of transfer data files that exist in the provided folder
    transfers = glob.glob(str(folder) + '/transfer_*')

    # Load the k-range data from the provided folder
    k_data = np.load(str(folder) + '/k_data.npy')

    # Initiate and empty dictionary where the transfer function data will be stored into
    transfer_dict = {}

    # Go through the transfer function list and read in the data individually
    for transfer in transfers:
        transfer_data = np.load(transfer)

        # Strip away the prefix and suffix to get the ell that this transfer function corresponds to
        ell = int(transfer.split('_')[1].split('.')[0])

        # Save the transfer function & k-data into the dictionary
        transfer_dict[ell] = [k_data, transfer_data]

    # Return the transfer_dict dictionary
    return transfer_dict
