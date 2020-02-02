# Created by Alessandro Maraio on 02/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is dedicated to performing the two- and three-point function integrals.
It will be expected that a lot of the details of the calculations will be abstracted away
from the user. Perhaps a few different integral methods could be used, which allows for cross-comparison
to ensure numerical accuracy and stability with speed comparisons too.
"""


class Integration:
    def __init__(self, transfer, database):
        # Set up the class, recording the database and transfer functions that will be used in the intergation
        self.transfer = transfer
        self.database = database
        self.type = self.database.type

    def integrate_power_spectrum(self):
        # Check that the database type is for a two-point function run and so can integrate the power spectrum
        if self.type != 'twopf':
            raise RuntimeError('Can not integrate the power spectrum on integration type that is not twopf.')

        # Import required modules for integration
        import numpy as np
        from scipy import integrate as sciint
        from scipy import interpolate as interp

        ell_list = self.transfer.get_ell_list()
        c_ell_list = []

        print('--- Starting two-point function integral ---')
        twopf_data = self.database.get_data()
        twopf_k_range = np.logspace(-6, 1.7, num=401)
        twopf_spline = interp.CubicSpline(twopf_k_range, twopf_data)

        # Build our integrand function for the SciPy integration.
        # Note: we normalise by 1E12 (which corresponds to units of micro-K^2) here to ensure that the
        # integrand values are not too tiny to cause numerical issues.
        def integrand(k, trans_spline, tpf_spline):
            return (4 * np.pi) * (1 / k) * (trans_spline(k) ** 2) * tpf_spline(k) * 1E12

        # Keep track on how long the integral takes using different methods.
        import time
        start_time = time.time()

        for index, ell in enumerate(ell_list):
            transfer_k, transfer_data = self.transfer.get_transfer(index)
            # Build a spline out of the transfer function. Very important that we have it sent to return zero
            # for values outside the interpolated region, otherwise this induces large numerical errors.
            # TODO: compare with other spline methods and/or libraries to see if performance and/or accuracy can be
            #  improved
            transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')

            # Define new integrand function for use with PyCuba integration routines
            def integrand_cuba(ndim, x_vec, ncomp, ff, userdata):
                kt = x_vec[0]
                ff[0] = integrand(kt, transfer_spline, twopf_spline)
                return 0

            # Booleans to set the integration method to be used
            # TODO: extract this away into function/class parameter
            UseScipy = True
            UseCuba = False

            if UseScipy:
                result, err = sciint.quad(integrand, 1E-6, 1.0, args=(transfer_spline, twopf_spline), epsabs=1E-12, epsrel=1E-12, limit=5000)
            elif UseCuba:
                import pycuba
                # Different Cuba routines are available. Currently only Vegas and Suave seem to work, not sure why??

                # results = pycuba.Cuhre(integrand_cuba, ndim=1, key=0, verbose=2, epsabs=1E-3)

                results = pycuba.Vegas(integrand_cuba, ndim=1, verbose=0, epsabs=1E-7)
                # results = pycuba.Suave(integrand_cuba, ndim=1, verbose=0, epsabs=1E-8, flatness=45)

                result = results['results'][0]['integral']
                error = results['results'][0]['error']

            result *= ell * (ell + 1) / (2 * np.pi)  # Multiply by common factor when using C_ell values

            c_ell_list.append(result)

        print('--- Finished two-point function integral ---')
        end_time = time.time()
        print('Time taken = ' + str(end_time - start_time))

        return ell_list, c_ell_list
