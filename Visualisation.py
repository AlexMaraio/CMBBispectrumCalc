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


def twopf_visualisation(ell, c_ell, style='seaborn', useLaTeX=False):
    # Default power spectrum plot of ell vs C_ell.
    import matplotlib.pyplot as plt
    if style == 'seaborn':
        import seaborn as sns
        sns.set(font_scale=2.25, rc={'text.usetex': True}) if useLaTeX else sns.set()

    plt.figure(figsize=(13, 7))
    plt.loglog(ell, c_ell, lw=2.5)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell \, \ell(\ell + 1)  / 2 \pi$ ')
    plt.title(r'Custom $C_\ell$ plot for quadratic $\frac{1}{2} m^2 \phi^2$')
    plt.show()
