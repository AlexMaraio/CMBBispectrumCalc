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


__all__ = [
    'cdist',
    'euclidean',
    'is_valid_dm',
    'minkowski',
    'wminkowski'
]


import warnings
import numpy as np

from functools import partial
from collections import namedtuple
from scipy._lib.six import callable
from scipy._lib.six import xrange

from scipy.spatial import _distance_wrap
from scipy.linalg import norm


def _args_to_kwargs_xdist(args, kwargs, metric, func_name):
    """
    Convert legacy positional arguments to keyword arguments for cdist.
    """
    if not args:
        return kwargs

    if callable(metric) and metric not in [euclidean, minkowski, wminkowski]:
        raise TypeError('When using a custom metric arguments must be passed as keyword (i.e., ARGNAME=ARGVALUE)')

    old_arg_names = ['p', 'V', 'VI', 'w']

    num_args = len(args)
    warnings.warn('%d metric parameters have been passed as positional.'
                  'This will raise an error in a future version.'
                  'Please pass arguments as keywords(i.e., ARGNAME=ARGVALUE)'
                  % num_args, DeprecationWarning)

    if num_args > 4:
        raise ValueError('Deprecated %s signature accepts only 4 positional arguments (%s), %d given.'
                         % (func_name, ', '.join(old_arg_names), num_args))

    for old_arg, arg in zip(old_arg_names, args):
        if old_arg in kwargs:
            raise TypeError('%s() got multiple values for argument %s' % (func_name, old_arg))
        kwargs[old_arg] = arg
    return kwargs


def _convert_to_type(x, out_type):
    return np.ascontiguousarray(x, dtype=out_type)


def _filter_deprecated_kwargs(kwargs, args_blacklist):
    # Filtering out old default keywords
    for k in args_blacklist:
        if k in kwargs:
            del kwargs[k]
            warnings.warn('Got unexpected kwarg %s. This will raise an error'
                          ' in a future version.' % k, DeprecationWarning)


def _validate_cdist_input(XA, XB, mA, mB, n, metric_name, **kwargs):
    if metric_name is not None:
        # get supported types
        types = _METRICS[metric_name].types
        # choose best type
        typ = types[types.index(XA.dtype)] if XA.dtype in types else types[0]
        # validate data
        XA = _convert_to_type(XA, out_type=typ)
        XB = _convert_to_type(XB, out_type=typ)

        # validate kwargs
        _validate_kwargs = _METRICS[metric_name].validator
        if _validate_kwargs:
            kwargs = _validate_kwargs(np.vstack([XA, XB]), mA + mB, n, **kwargs)
    else:
        typ = None
    return XA, XB, typ, kwargs


def _validate_minkowski_kwargs(X, m, n, **kwargs):
    if 'p' not in kwargs:
        kwargs['p'] = 2.
    return kwargs


def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def _validate_weights(w, dtype=np.double):
    w = _validate_vector(w, dtype=dtype)
    if np.any(w < 0):
        raise ValueError("Input weights should be all non-negative")
    return w


def _validate_wminkowski_kwargs(X, m, n, **kwargs):
    w = kwargs.pop('w', None)
    if w is None:
        raise ValueError('weighted minkowski requires a weight '
                         'vector `w` to be given.')
    kwargs['w'] = _validate_weights(w)
    if 'p' not in kwargs:
        kwargs['p'] = 2.
    return kwargs


def minkowski(u, v, p=2, w=None):
    """
    Compute the Minkowski distance between two 1-D arrays.

    The Minkowski distance between 1-D arrays `u` and `v`,
    is defined as

    .. math::

       {||u-v||}_p = (\\sum{|u_i - v_i|^p})^{1/p}.


       \\left(\\sum{w_i(|(u_i - v_i)|^p)}\\right)^{1/p}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    p : int
        The order of the norm of the difference :math:`{||u-v||}_p`.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    minkowski : double
        The Minkowski distance between vectors `u` and `v`.

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if p < 1:
        raise ValueError("p must be at least 1")
    u_v = u - v
    if w is not None:
        w = _validate_weights(w)
        if p == 1:
            root_w = w
        if p == 2:
            # better precision and speed
            root_w = np.sqrt(w)
        else:
            root_w = np.power(w, 1/p)
        u_v = root_w * u_v
    dist = norm(u_v, ord=p)
    return dist


# `minkowski` gained weights in scipy 1.0.  Once we're at say version 1.3,
# deprecated `wminkowski`.  Not done at once because it would be annoying for
# downstream libraries that used `wminkowski` and support multiple scipy
# versions.
def wminkowski(u, v, p, w):
    """
    Compute the weighted Minkowski distance between two 1-D arrays.

    The weighted Minkowski distance between `u` and `v`, defined as

    .. math::

       \\left(\\sum{(|w_i (u_i - v_i)|^p)}\\right)^{1/p}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    p : int
        The order of the norm of the difference :math:`{||u-v||}_p`.
    w : (N,) array_like
        The weight vector.

    Returns
    -------
    wminkowski : double
        The weighted Minkowski distance between vectors `u` and `v`.

    Notes
    -----
    `wminkowski` is DEPRECATED. It implements a definition where weights
    are powered. It is recommended to use the weighted version of `minkowski`
    instead. This function will be removed in a future version of scipy.

    """
    w = _validate_weights(w)
    return minkowski(u, v, p=p, w=w**p)


def euclidean(u, v, w=None):
    """
    Computes the Euclidean distance between two 1-D arrays.

    The Euclidean distance between 1-D arrays `u` and `v`, is defined as

    .. math::

       {||u-v||}_2

       \\left(\\sum{(w_i |(u_i - v_i)|^2)}\\right)^{1/2}

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    euclidean : double
        The Euclidean distance between vectors `u` and `v`.

    """
    return minkowski(u, v, p=2, w=w)


_convert_to_double = partial(_convert_to_type, out_type=np.double)
_convert_to_bool = partial(_convert_to_type, out_type=bool)


# Registry of implemented metrics:
# Dictionary with the following structure:
# {
#  metric_name : MetricInfo(aka, types=[double], validator=None)
# }
#
# Where:
# `metric_name` must be equal to python metric name
#
# MetricInfo is a named tuple with fields:
#  'aka' : [list of aliases],
#
#  'validator': f(X, m, n, **kwargs)    # function that check kwargs and
#                                       # computes default values.
#
#  'types': [list of supported types],  # X (pdist) and XA (cdist) are used to
#                                       # choose the type. if there is no match
#                                       # the first type is used. Default double
# }
MetricInfo = namedtuple("MetricInfo", 'aka types validator ')
MetricInfo.__new__.__defaults__ = (['double'], None)

_METRICS = {
    'euclidean': MetricInfo(aka=['euclidean', 'euclid', 'eu', 'e']),
    'minkowski': MetricInfo(aka=['minkowski', 'mi', 'm', 'pnorm'],
                            validator=_validate_minkowski_kwargs),
    'wminkowski': MetricInfo(aka=['wminkowski', 'wmi', 'wm', 'wpnorm'],
                             validator=_validate_wminkowski_kwargs)
}

_METRIC_ALIAS = dict((alias, name)
                     for name, info in _METRICS.items()
                     for alias in info.aka)

_METRICS_NAMES = list(_METRICS.keys())

_TEST_METRICS = {'test_' + name: globals()[name] for name in _METRICS.keys()}


def _select_weighted_metric(mstr, kwargs, out):
    kwargs = dict(kwargs)

    if "w" in kwargs and kwargs["w"] is None:
        # w=None is the same as omitting it
        kwargs.pop("w")

    if mstr.startswith("test_") or mstr in _METRICS['wminkowski'].aka:
        # These support weights
        pass
    elif "w" in kwargs:
        if mstr in _METRICS['seuclidean'].aka:
            raise ValueError("metric %s incompatible with weights" % mstr)

        # XXX: C-versions do not support weights
        # need to use python version for weighting
        kwargs['out'] = out
        mstr = "test_%s" % mstr

    return mstr, kwargs


def is_valid_dm(D, tol=0.0, throw=False, name="D", warning=False):
    """
    Return True if input array is a valid distance matrix.

    Distance matrices must be 2-dimensional numpy arrays.
    They must have a zero-diagonal, and they must be symmetric.

    Parameters
    ----------
    D : ndarray
        The candidate object to test for validity.
    tol : float, optional
        The distance matrix should be symmetric. `tol` is the maximum
        difference between entries ``ij`` and ``ji`` for the distance
        metric to be considered symmetric.
    throw : bool, optional
        An exception is thrown if the distance matrix passed is not valid.
    name : str, optional
        The name of the variable to checked. This is useful if
        throw is set to True so the offending variable can be identified
        in the exception message when an exception is thrown.
    warning : bool, optional
        Instead of throwing an exception, a warning message is
        raised.

    Returns
    -------
    valid : bool
        True if the variable `D` passed is a valid distance matrix.

    Notes
    -----
    Small numerical differences in `D` and `D.T` and non-zeroness of
    the diagonal are ignored if they are within the tolerance specified
    by `tol`.

    """
    D = np.asarray(D, order='c')
    valid = True
    try:
        s = D.shape
        if len(D.shape) != 2:
            if name:
                raise ValueError(('Distance matrix \'%s\' must have shape=2 '
                                  '(i.e. be two-dimensional).') % name)
            else:
                raise ValueError('Distance matrix must have shape=2 (i.e. '
                                 'be two-dimensional).')
        if tol == 0.0:
            if not (D == D.T).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' must be '
                                     'symmetric.') % name)
                else:
                    raise ValueError('Distance matrix must be symmetric.')
            if not (D[xrange(0, s[0]), xrange(0, s[0])] == 0).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' diagonal must '
                                      'be zero.') % name)
                else:
                    raise ValueError('Distance matrix diagonal must be zero.')
        else:
            if not (D - D.T <= tol).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' must be '
                                      'symmetric within tolerance %5.5f.')
                                     % (name, tol))
                else:
                    raise ValueError('Distance matrix must be symmetric within'
                                     ' tolerance %5.5f.' % tol)
            if not (D[xrange(0, s[0]), xrange(0, s[0])] <= tol).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' diagonal must be'
                                      ' close to zero within tolerance %5.5f.')
                                     % (name, tol))
                else:
                    raise ValueError(('Distance matrix \'%s\' diagonal must be'
                                      ' close to zero within tolerance %5.5f.')
                                     % tol)
    except Exception as e:
        if throw:
            raise
        if warning:
            warnings.warn(str(e))
        valid = False
    return valid


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

    Raises
    ------
    ValueError
        An exception is thrown if `XA` and `XB` do not have
        the same number of columns.
    """
    kwargs = dict(kwargs)

    dm = np.empty((x_a.shape[0], x_b.shape[0]), dtype=np.double)

    x_a = np.ascontiguousarray(x_a, dtype='double')
    x_b = np.ascontiguousarray(x_b, dtype='double')

    _distance_wrap.cdist_euclidean_double_wrap(x_a, x_b, dm, **kwargs)
    return dm
