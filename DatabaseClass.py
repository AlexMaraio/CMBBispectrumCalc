# Created by Alessandro Maraio on 01/02/2020.
# Copyright (c) 2020 University of Sussex.
# contributor: Alessandro Maraio <am963@sussex.ac.uk>


"""
This file is dedicated to the handling of CppTransport SQLite databases and extracting the
necessary information out of them for use with the two- and three-point CMB integrals.
"""

import sqlite3


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

    def get_data(self):
        if self.type == 'twopf':
            with self.conn:
                cur = self.conn.cursor()
                cur.execute("SELECT twopf_samples.*, zt.tserial, zt.twopf "
                            "FROM twopf_samples "
                            "INNER JOIN zeta_twopf zt on twopf_samples.serial = zt.kserial "
                            "WHERE tserial = (SELECT MAX(tserial) FROM zeta_twopf) "
                            "ORDER BY serial")

                rows = cur.fetchall()

                data = []
                for row in rows:
                    data.append(row[6])

            return data
