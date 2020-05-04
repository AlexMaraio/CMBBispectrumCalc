# Created by Alessandro Maraio on 02/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file will be where our visualisation tools will be put so that way
we can present the results for the two- and three-point function integrals
in a nice and coherent way.

This will be especially true for the three-point function because it is a function
of 3 variables, we will need to plot on a 3D grid and so some work will need to get this right.
"""


import matplotlib.pyplot as plt


def twopf_visualisation(ell, c_ell, use_seaborn=True, use_LaTeX=False):
    """
    Function for creating a default power spectrum plot for C_ell vs ell, in standard plotting units.

    Takes ell and c_ell, which are lists that corresponds to the integration data, and optional bool switches
    on or off using the seaborn plotting style, and using LaTeX labels.
    """

    # If using seaborn, import it and set plot options given function inputs
    if use_seaborn:
        import seaborn as sns
        sns.set(font_scale=1.75, rc={'text.usetex': True}) if use_LaTeX else sns.set(font_scale=1.5)

    # Create a matplotlib figure and plot the power spectrum on it.
    plt.figure(figsize=(16, 9))
    plt.loglog(ell, c_ell, lw=2.5)
    plt.xlabel(r'$\ell$')

    # Change the y-label depending on if we're using LaTeX labels or not
    if use_LaTeX:
        plt.ylabel(r'$C_\ell \,\, \ell(\ell + 1)  / 2 \pi \,\, [\mu \textrm{K}^2]$')
    else:
        plt.ylabel(r'$C_\ell \,\, \ell(\ell + 1)  / 2 \pi \,\, [\mu K^2]$')

    plt.title(r'CMB TT power spectrum $C_\ell$ plot')
    plt.tight_layout()
    plt.show()


def equal_ell_bispectrum_plot(ell, bispec, use_seaborn=True, use_LaTeX=False, save_folder=None):
    """
    Function that creates a default plot of the equal ell CMB bispectrum.

    Takes arguments of the ell and bispec, which are lists that corresponds to the integration data,
    two bool switches that determine plot styling options,
    and an optional string save_folder, which if provided saves the figure into the folder given.
    """

    # Import seaborn if we are using it and initialise with plot settings determined by function inputs
    if use_seaborn:
        import seaborn as sns
        sns.set(font_scale=1.75, rc={'text.usetex': True}) if use_LaTeX else sns.set(font_scale=1.5)

    # Create a matplotlib figure and plot the bispectrum on it
    plt.figure(figsize=(16, 9))
    plt.semilogx(ell, bispec, lw=2.5)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'Bispectrum $b_{\ell \, \ell \, \ell}$')
    plt.title(r'CMB TTT bispectrum plot for $\ell_1 = \ell_2 = \ell_3 \equiv \ell$')
    plt.tight_layout()

    # If a save folder is provided, save the figure to that folder.
    if save_folder is not None:
        plt.savefig(str(save_folder) + '/bispectrum_equal_ell.png', dpi=500)

    plt.show()
