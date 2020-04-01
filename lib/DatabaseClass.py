# Created by Alessandro Maraio on 01/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is dedicated to the handling of CppTransport SQLite databases and extracting the
necessary information out of them for use with the two- and three-point CMB integrals.

We are using the pandas library for native integration between the SQLite databases
and Python.
"""


import os
import sqlite3
import pandas as pd
from scipy import interpolate as interp


class Database:

    def __init__(self, filepath, db_type):
        self.conn = None

        # Check that the provided database type is either 'twopf' or 'threepf'. If not, raise error
        if db_type not in ['twopf', 'threepf']:
            raise RuntimeError('Incorrect database type. Please check and re-run.')

        self.type = db_type.lower()

        try:
            self.conn = sqlite3.connect(filepath)
        except sqlite3.Error as e:
            print(e)

        self.k_table = None

    def __exit__(self, *args):
        self.conn.close()

    def get_dataframe(self):
        if self.type == 'twopf':
            df = pd.read_sql_query("""
                                    SELECT twopf_samples.*, zt.tserial, times.time,  zt.twopf 
                                    FROM twopf_samples 
                                    INNER JOIN zeta_twopf zt on twopf_samples.serial = zt.kserial 
                                    INNER JOIN time_samples times on zt.tserial = times.serial
                                    WHERE tserial = (SELECT MAX(tserial) FROM zeta_twopf) 
                                    ORDER BY serial
                                    """, self.conn)

            return df

        elif self.type == 'threepf':
            df = pd.read_sql_query("""
                                    SELECT t1.conventional AS k1,
                                    t2.conventional AS k2,
                                    t3.conventional AS k3,
                                    zt.threepf
                                    FROM threepf_samples
                                    INNER JOIN zeta_threepf zt on threepf_samples.serial = zt.kserial
                                    INNER JOIN time_samples ts on zt.tserial = ts.serial
                                    INNER JOIN twopf_samples t1 on threepf_samples.wavenumber1 = t1.serial
                                    INNER JOIN twopf_samples t2 on threepf_samples.wavenumber2 = t2.serial
                                    INNER JOIN twopf_samples t3 on threepf_samples.wavenumber3 = t3.serial
                                    WHERE tserial = (SELECT MAX(tserial) FROM zeta_threepf);
                                    """, self.conn)

            return df

    def set_k_table(self, k_table_path):
        # When running a CppTransport CMB task, a k_table.dat file is created which allows for the conversion
        # between CppT internal k values, and physical k values.
        # This function checks that the provided filepath exists, and if not raises an error

        if os.path.isfile(k_table_path):
            self.k_table = k_table_path

        else:
            raise RuntimeError('k_table_path is not a valid file, please ensure that it points to the k_table.dat '
                               'file created by CppTransport and run again.')

    def save_inflation_bispec(self, folder):
        # This function reads in the provided inflationary bispectrum and k_table file to transform the
        # internal CppTransport k values into physical k values. This then saves the new dataframe, which can then
        # be read in by the individual workers.

        # Ensures that k_table exists
        if self.k_table is None:
            raise RuntimeError('Please set the path to the k_table first and then re-run this function')

        print('--- Reading in inflationary bispectrum data ---', flush=True)

        # Gets the inflationary bispectrum data in form of dataframe
        data = self.get_dataframe()

        # Reads in the k_data file, and stores in a dataframe
        k_data = pd.read_csv(str(self.k_table), sep='\t')

        # Uses SciPy interpolate CubicSpline to construct a spline between CppT and physical k values
        k_spline = interp.CubicSpline(k_data['k_cpptransport'], k_data['k_physical'])

        # Transform the k values to physical k values using the spline constructed above
        data['k1'] = k_spline(data['k1'])
        data['k2'] = k_spline(data['k2'])
        data['k3'] = k_spline(data['k3'])

        # Saves the data to the provided save_folder
        data.to_csv(str(folder) + '/inflationary_bispectrum_data.csv', index=False)

        print('--- Inflationary bispectrum data saved to save_folder ---', flush=True)

        # Manually delete variables now that the data has been saved, to try to help manual memory management
        del data, k_data, k_spline