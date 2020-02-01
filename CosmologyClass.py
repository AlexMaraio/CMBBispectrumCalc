# Created by Alessandro Maraio on 31/01/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>



class Cosmology:
    """ 
    The Cosmology class creates a new instance that holds all of the 
    cosmological information about the model that we are interested in.
    
    self is meant to help abstarct away dependance on the Bolzmann code
    by specalising the specific cosmology here, which can then be passed
    onto the different codes in their own way.
    """
    
    def __init__(self, template = 'Planck2018'):
        # Creates new class of cosmology, using the template
        
        template = template.lower() # Ensure that template is in lower form
        
        if (template == "planck 2018") or (template == "planck2018") or (template == "planck18"):
            self.template = "Planck 2018"
            self.H0 = 67.36
            self.omega_bh2 = 0.02237
            self.omega_cdmh2 = 0.1200
            self.tau = 0.0544

        elif (template == "planck 2015" or template == "planck2015" or template == "planck15"):
            self.template = "Planck 2015"
            self.H0 = 67.51
            self.omega_bh2 = 0.02226
            self.omega_cdmh2 = 0.1193
            self.tau = 0.063

  
        elif (template == "planck 2013" or template == "planck2013" or template == "planck13"):
            self.template = "Planck 2013"
            self.H0 = 67.9
            self.omega_bh2 = 0.02218
            self.omega_cdmh2 = 0.1186
            self.tau = 0.090

        elif (template == "wmap 9" or template == "wmap9"):
            self.template = "WMAP 9"
            self.H0 = 70.0
            self.omega_bh2 = 0.02264
            self.omega_cdmh2 = 0.1138
            self.tau = 0.089
            
        else:
            raise RuntimeError('Invalid cosmology template used')
        
        
    
