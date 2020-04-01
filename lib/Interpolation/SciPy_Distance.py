"""
Distance computations (:mod:`scipy.spatial.distance`)
=====================================================

.. sectionauthor:: Damian Eads

Function Reference
------------------

Distance matrix computation from a collection of raw observation vectors
stored in a rectangular array.

   cdist   -- distances between two collections of observation vectors
"""
# Copyright (C) Damian Eads, 2007-2008. New BSD License.


from __future__ import division, print_function, absolute_import


__all__ = ['cdist']


import numpy as np

from scipy.spatial import _distance_wrap


def cdist(x_a, x_b, metric='euclidean', *args, **kwargs):
    """
    Compute distance between each pair of the two collections of inputs.

    Parameters
    ----------
    x_a : ndarray
        An :math:`m_A` by :math:`n` array of :math:`m_A`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    x_b : ndarray
        An :math:`m_B` by :math:`n` array of :math:`m_B`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    metric : str

    Returns
    -------
    Y : ndarray
        A :math:`m_A` by :math:`m_B` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.

    """
    kwargs = dict(kwargs)

    dm = np.empty((x_a.shape[0], x_b.shape[0]), dtype=np.double)

    x_a = np.ascontiguousarray(x_a, dtype='double')
    x_b = np.ascontiguousarray(x_b, dtype='double')

    _distance_wrap.cdist_euclidean_double_wrap(x_a, x_b, dm, **kwargs)
    return dm
