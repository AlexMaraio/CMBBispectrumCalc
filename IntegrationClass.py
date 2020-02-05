# Created by Alessandro Maraio on 02/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is dedicated to performing the two- and three-point function integrals.
It will be expected that a lot of the details of the calculations will be abstracted away
from the user. Perhaps a few different integral methods could be used, which allows for cross-comparison
to ensure numerical accuracy and stability with speed comparisons too.
"""


def integrate(args):
    import numpy as np
    from scipy import interpolate as interp
    from scipy import integrate as sciint

    transfer_k = args[1]
    transfer_data = args[2]
    twopf_spline = args[3]

    transfer_spline = interp.InterpolatedUnivariateSpline(transfer_k, transfer_data, ext='zeros')

    def log_integrand(k, el, trans_spline, tpf_spline):
        return 2 * (trans_spline(np.exp(k)) ** 2) * tpf_spline(np.exp(k)) * 1E12 * el * (el + 1)

    result, err = sciint.quad(log_integrand, -15, 4, args=(args[0], transfer_spline, twopf_spline), epsabs=1E-4,
                              epsrel=1E-4, limit=5000)
    return result

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

        def log_integrand(k, l, trans_spline, tpf_spline):
            return 2 * (trans_spline(np.exp(k)) ** 2) * tpf_spline(np.exp(k)) * 1E12 * l * (l + 1)  # (4 * np.pi)

        def ode_integrand(k, c_ell,  trans_spline, tpf_spline):
            return [(4 * np.pi) * (1 / k) * (trans_spline(k) ** 2) * tpf_spline(k) * 1E12]

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
            UseScipyQuad    = True
            UseScipyOdeint  = False
            UseCuba         = False

            '''
            k_plot = np.logspace(-6, 1, 5000)
            import matplotlib.pyplot as plt
            integrand_plot = []
            for k_thing in k_plot:
                integrand_plot.append(integrand(k_thing, transfer_spline, twopf_spline))

            plt.loglog(k_plot, integrand_plot)
            plt.title(r'$\l =$ ' + str(l))
            plt.show()
            '''

            if UseScipyQuad:
                # result, err = sciint.quad(integrand, 1E-4, 1, args=(transfer_spline, twopf_spline), epsabs=1E-10, epsrel=1E-10, limit=5000)
                result, err = sciint.quad(log_integrand, -15, 4, args=(ell, transfer_spline, twopf_spline), epsabs=1E-4, epsrel=1E-4, limit=5000)
                #  result, err = sciint.quadrature(integrand, 1E-6, 1.0, args=(transfer_spline, twopf_spline),
                #  maxiter=5000)

            elif UseScipyOdeint:
                solution = sciint.solve_ivp(ode_integrand, [1E-6, 1], y0=[0], args=(transfer_spline, twopf_spline), dense_output=True, method='BDF', atol=1E-10, rtol=1E-10)
                #  print(solution)
                #  result = solution.y[-1]
                result = solution.sol(0.8)
                #  print(result)
            elif UseCuba:
                import pycuba
                # Different Cuba routines are available. Currently only Vegas and Suave seem to work, not sure why??

                results = pycuba.Cuhre(integrand_cuba, ndim=2, verbose=0, epsabs=1E-8, epsrel=1E-8)
                # results = pycuba.Vegas(integrand_cuba, ndim=1, verbose=0, epsabs=1E-6)
                # results = pycuba.Suave(integrand_cuba, ndim=1, verbose=0, epsabs=1E-6, flatness=75)

                result = results['results'][0]['integral']
                error = results['results'][0]['error']

            #result *= l * (l + 1) / (2 * np.pi)  # Multiply by common factor when using C_ell values

            c_ell_list.append(result)

        print('--- Finished two-point function integral ---')
        end_time = time.time()
        print('Time taken = ' + str(end_time - start_time))

        return ell_list, c_ell_list

    def integrate_power_spectrum_parallel(self):
        # Check that the database type is for a two-point function run and so can integrate the power spectrum
        if self.type != 'twopf':
            raise RuntimeError('Can not integrate the power spectrum on integration type that is not twopf.')

        # Import required modules for integration
        import numpy as np
        from scipy import integrate as sciint
        from scipy import interpolate as interp
        import multiprocessing as multi
        #from multiprocessing import Pool

        ell_list = self.transfer.get_ell_list()
        c_ell_list = []

        print('--- Starting two-point function integral ---')
        twopf_data = self.database.get_data()
        twopf_k_range = np.logspace(-6, 1.7, num=401)
        twopf_spline = interp.CubicSpline(twopf_k_range, twopf_data)


        # Keep track on how long the integral takes using different methods.
        import time
        start_time = time.time()

        big_list = []

        for index, ell in enumerate(ell_list):
            transfer_k, transfer_data = self.transfer.get_transfer(index)
            temp = [ell, transfer_k, transfer_data, twopf_spline ]
            big_list.append(temp)


        pool = multi.Pool(8)

        c_ell_list = pool.map(integrate, big_list)



        print('--- Finished two-point function integral ---')
        end_time = time.time()
        print('Time taken = ' + str(end_time - start_time))

        return ell_list, c_ell_list

    def integrate_power_spectrum_cpp(self):
        ell_list = self.transfer.get_ell_list()
        c_ell_list = []

        import time
        start_time = time.time()

        # Import required modules for integration
        import numpy as np
        import os
        import ctypes
        from scipy import integrate as sciint
        from scipy import LowLevelCallable

        lib = ctypes.CDLL(os.path.abspath('./cpp/build/libCppIntegrand.so'))
        lib.get_twopf_data.restype = ctypes.c_double
        lib.get_transfer_data.restype = ctypes.c_double
        lib.get_integrand.restype = ctypes.c_double
        lib.get_integrand.argtypes = [ctypes.c_double]
        lib.get_log_integrand.restype = ctypes.c_double
        lib.get_log_integrand.argtypes = [ctypes.c_double, ctypes.c_void_p]

        print('--- Starting two-point function integral ---')
        twopf_k_range = np.logspace(-6, 1.7, num=401, dtype=float)
        twopf_data = np.array(self.database.get_data(), dtype=float)

        twopf_k_range_ctype = twopf_k_range.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        twopf_data_ctype = twopf_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        lib.set_twopf_data(twopf_k_range_ctype, twopf_data_ctype, len(twopf_data))

        '''
        twopf_list = []
        for k in twopf_k_range:
            p_k = ctypes.c_double(k)
            twopf_list.append(lib.get_twopf_data(p_k))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(twopf_k_range, twopf_data)
        plt.show(block=True)
        '''

        #func = LowLevelCallable(lib.get_log_integrand)

        for index, ell in enumerate(ell_list):
            transfer_k, transfer_data = self.transfer.get_transfer(index)

            transfer_k = np.array(transfer_k, dtype=float)
            transfer_data = np.array(transfer_data, dtype=float)
            transfer_k_ctype = transfer_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            transfer_data_ctype = transfer_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            lib.set_transfer_data(transfer_k_ctype, transfer_data_ctype, len(transfer_k))

            ell_c = ctypes.c_double(ell)
            user_data = ctypes.cast(ctypes.pointer(ell_c), ctypes.c_void_p)

            func = LowLevelCallable(lib.get_log_integrand, user_data)

            k_list = np.logspace(-15, 4, num=8024)
            integrand_list = []
            for k in k_list:
                c_k = ctypes.c_double(k)
                integrand_list.append(lib.get_integrand(c_k))
            """
            import matplotlib.pyplot as plt
            plt.figure()
            plt.loglog(k_list, integrand_list)
            plt.title(r'$\ell$ = ' + str(ell))
            plt.show(block=True)
            """

            # result, err = sciint.quad(func, -10, 1, epsabs=1E-18, epsrel=1E-18, limit=5000)

            # result, err = sciint.nquad(func, [[1E-4, 0.1]], opts={'epsabs': 1E-16, 'epsrel': 1E-16, 'limit': 5000})
            # result = sciint.romb(integrand_list, k_list)

            # result = lib.Do_Cuba_Integration()

            # result *= ell * (ell + 1) / (2 * np.pi)  # Multiply by common factor when using C_ell values
            result = 1
            c_ell_list.append(result)

        print('--- Finished two-point function integral ---')
        end_time = time.time()
        print('Time taken = ' + str(end_time - start_time))

        return ell_list, c_ell_list

