# Created by Alessandro Maraio on 01/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is dedicated to the handling of CppTransport SQLite databases and extracting the
necessary information out of them for use with the two- and three-point CMB integrals.

We are using the pandas library for native integration between the SQLite databases
and Python.
"""

import sqlite3
import pandas as pd


class Database:

    def __init__(self, filepath, db_type):
        self.conn = None
        self.type = db_type.lower()

        try:
            self.conn = sqlite3.connect(filepath)
        except sqlite3.Error as e:
            print(e)

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
