# Created by Alessandro Maraio on 31/01/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is designed to implement the interface to a Boltzmann code
that is used to calculate the transfer functions of the temperature 
anisotropy of the CMB.

The goal here is to establish which code we are using (either CAMB
or CLASS) and then provide a consistant interface for the main code
as here we can deal with the specalisms of each code
""" 



class BolzmannCode:
    
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
		              
            self.params = camb.CAMBparams()
            self.params.set_cosmology(H0 = cosmology.H0, ombh2 = cosmology.omega_bh2, \
						             omch2 = cosmology.omega_cdmh2, tau = cosmology.tau)
				
			
		
		
        elif code.lower() == 'class':
	        raise RuntimeError('CLASS is currently not supported, please use CAMB instead.')
	        
        else:
	        raise RuntimeError('Unkown Bolzmann code specified.')
	





