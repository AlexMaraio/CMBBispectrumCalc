# Created by Alessandro Maraio on 01/04/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file hosts the class Grid, which is responsible for generating the correct (ell1, ell2, ell3) grid
for use in the bispectrum integration
"""

import numpy as np
import pandas as pd


class Grid:
    def __init__(self):
        """
        Class constructor which needs no arguments.

        Initialises  all class parameters to None
        """

        # (pandas DataFrame): DataFrame object that holds the (ell1, ell2, ell3) configuration list
        self.grid = None

        # (str): The grid type that we are going to build
        self.type = None

        # (int): The maximum ell that the grid has used to construct the grid
        self.ell_max = None

        # (int): The step that has been used to construct the grid
        self.ell_step = None

    def build_equal_ell_grid(self, ell_max=2500, ell_step=5):
        """
        Builds the grid of allowed ell values, that satisfy the equal-ell condition of ell1 = ell2 = ell3

        This is useful for plotting the bispectrum as a function of only one variable

        Args:
            ell_max: The maximum ell value that the grid is constructed up to
            ell_step: The individual ell step in the grid

        Returns:
            None: The function saves the pandas grid to the class, and so does not return anything
        """

        print('--- Building grid of ells ---', flush=True)

        # Set class parameters for this type of ell grid
        self.type = 'equal_ell_grid'
        self.ell_max = ell_max
        self.ell_step = ell_step

        # Construct an ell list, which goes up the the specified maximum ell in the specified ell steps
        ell_list = np.arange(30, ell_max, ell_step)

        # Initialise an empty list which is where accepted ell configurations will go into
        allowed_ells = []

        for ell in ell_list:
            temp = {'index': len(allowed_ells), 'ell1': ell, 'ell2': ell+15, 'ell3': ell-15}
            allowed_ells.append(temp)

        # Convert the list of dictionaries to a pandas DataFrame
        allowed_ells = pd.DataFrame(allowed_ells)

        # Print configuration statistics
        print('--- Built grid of ells ---', flush=True)
        print('--- Number of ell configurations ' + str(allowed_ells.shape[0]) + ' ---', flush=True)

        # Save the built data grid to the class
        self.grid = allowed_ells

    def build_ell_sum_grid(self, ell_sum=4000, ell_max=1700, ell_step=10):
        """
        Builds a grid of allowed [ell1, ell2, ell3] values that sum to produce ell_sum, with an individual maximum
        ell value of ell_cut, built using steps of ell_step in the individual values.

        Args:
            ell_sum (int): This is the value of ell1 + ell2 + ell3 that the grid will be constructed to fit
            ell_max (int): This is the maximum ell value that will be used to construct the grid
            ell_step (int): This the individual ell step that the grid will be made out of

        Returns:
            None: This function saves the built grid to the class, and so does not return anything
        """

        # If the value of ell1 + ell2 + ell3 is not even, then raise an error as this is an invalid grid
        if ell_sum % 2 != 0:
            raise RuntimeError('The value of ell_sum must be an even number for the bispectrum to be non-zero.')

        print('--- Building grid of ells ---', flush=True)

        # Set class parameters for this type of ell grid
        self.type = 'ell_sum_grid'
        self.ell_max = ell_max  # TODO: change this asap!
        self.ell_step = ell_step

        # Construct an ell list, which goes up the the specified maximum ell in the specified ell steps
        ell_list = np.arange(10, ell_max, ell_step)

        # Initialise an empty list which is where accepted ell configurations will go into
        allowed_ells = []

        # Loop through each ell list
        for ell1 in ell_list:
            for ell2 in ell_list:
                for ell3 in ell_list:
                    # Test to see if the ell values match the specified ell sum
                    if ell1 + ell2 + ell3 == ell_sum:
                        # If it does, add this configuration to the list of accepted values
                        temp = {'index': len(allowed_ells), 'ell1': ell1, 'ell2': ell2, 'ell3': ell3}
                        allowed_ells.append(temp)

        allowed_ells = pd.DataFrame(allowed_ells)

        # If the returned list is empty, then we did not have any valid configurations, so raise an error
        if allowed_ells.shape[0] == 0:
            raise RuntimeError('The ell grid that was specified has no valid triangle configurations for parameters \n'
                               'ell_sum: ' + str(ell_sum) + ', ell_step: ' + str(ell_step) + ', ell_max: ' +
                               str(ell_max) +
                               '\nPlease try again with more relaxed values (decreased ell_step, increased ell_max)')

        # Print configuration statistics
        print('--- Built grid of ells ---', flush=True)
        print('--- Number of ell configurations ' + str(allowed_ells.shape[0]) + ' ---', flush=True)

        # Save the built data grid to the class
        self.grid = allowed_ells

    def build_ell_volume_grid(self, ell_max=2000, ell_step=10, full_grid=True):
        """
        Builds a grid of allowed (ell1, ell2, ell3) values that are allowed by the ell selection rules in place
        for the bispectrum configurations.
        We build the grid in steps using ell_steps and up to an individual ell maximum of ell_max.

        The selection rules are:
            - Parity condition: ell1 + ell2 + ell3 = even
            - Triangle condition: ell1, ell2, ell3 must form a triangle from their values

        Args:
            ell_max (int): The maximum ell that the grid will be built out of
            ell_step (int): The steps that each ell will be constructed out of
            full_grid (bool): Default is True, which builds the full (ell1, ell2, ell3) gird of allowed configurations.
                              If False, then the condition that ell3 > ell2 > ell1 is imposed.

        Returns:
            None: This function saves the built grid to the class, and so does not return anything
        """

        print('--- Building grid of ells ---', flush=True)

        # Set class properties for the type of grid that is being constructed
        self.type = 'ell_volume_grid'
        self.ell_max = ell_max
        self.ell_step = ell_step

        # Construct each individual ell list from 10 to ell_max in steps of ell_step
        ell_list = np.arange(10, ell_max, ell_step)

        # Initialise an empty list which is where accepted ell configurations will go into
        allowed_ells = []

        # Loop through each ell list individually
        for ell1 in ell_list:
            for ell2 in ell_list:
                for ell3 in ell_list:

                    # If we are not building a full grid, then build a condensed grid where ell1 < ell2 < ell3
                    if not full_grid:
                        if ell1 < ell2 < ell3:
                            continue

                    # First validity check, ensure that ell1 + ell2 + ell3 is even
                    if (ell1 + ell2 + ell3) % 2 != 0:
                        continue

                    # Then check that the ell values can form a valid triangle
                    if (ell1 + ell2 <= ell3) or (ell1 + ell3 <= ell2) or (ell2 + ell3 <= ell1):
                        continue

                    # If it's passed, then append the configuration to the accepted list
                    allowed_ells.append({'index': len(allowed_ells), 'ell1': ell1, 'ell2': ell2, 'ell3': ell3})

        # Convert the list to a pandas DataFra,e
        allowed_ells = pd.DataFrame(allowed_ells)

        # If the list is empty, then no valid configurations have been found, so raise an error
        if allowed_ells.shape[0] == 0:
            raise RuntimeError('The ell grid that was specified has no valid triangle configurations for parameters \n'
                               'ell_step: ' + str(ell_step) + ', ell_max: ' + str(ell_max) + '\n'
                               'Please try again with more relaxed values (decreased ell_step, increased ell_max)')

        # Print summary statistics
        print('--- Built grid of ells ---', flush=True)
        print('--- Number of ell configurations ' + str(allowed_ells.shape[0]) + ' ---', flush=True)

        # Save the built grid into the class
        self.grid = allowed_ells

    @property
    def get_grid(self):
        """
        Function to access the ell grid that has been constructed in the class

        Returns:
            self.grid (pandas DataFrame): Returns the ell configurations in a pandas DataFrame
        """

        # Ensures that the grid has been built first
        if self.grid is None:
            raise RuntimeError('No grid has been constructed yet! Please generate one using either: build_ell_sum_grid '
                               'or build_ell_volume_grid.')

        # Then returns the grid
        return self.grid
