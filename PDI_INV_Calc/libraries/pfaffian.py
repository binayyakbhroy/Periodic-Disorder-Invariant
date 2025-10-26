import numpy as np
import scipy.linalg as la
from pfapack.ctypes import pfaffian as cpf
from scipy import sparse
from scipy.linalg import ishermitian

def sign(x):
    signed_x = np.zeros_like(x)
    for x0 in x:
        if x0 > 0:
            signed_x[x == x0] = 1
        elif x0 < 0:
            signed_x[x == x0] = -1
        else:
            signed_x[x == x0] = 0
    return signed_x

def pfaffian_schur(A, overwrite_a=False):
    """Calculate Pfaffian of a real antisymmetric matrix using
    the Schur decomposition. (Hessenberg would in principle be faster,
    but scipy-0.8 messed up the performance for scipy.linalg.hessenberg()).

    This function does not make use of the skew-symmetry of the matrix A,
    but uses a LAPACK routine that is coded in FORTRAN and hence faster
    than python. As a consequence, pfaffian_schur is only slightly slower
    than pfaffian().
    """

    assert np.issubdtype(A.dtype, np.number) and not np.issubdtype(
        A.dtype, np.complexfloating
    )

    assert A.shape[0] == A.shape[1] > 0

    assert abs(A + A.T).max() < 1e-14

    # Quick return if possible
    if A.shape[0] % 2 == 1:
        return 0

    (t, z) = la.schur(A, output="real", overwrite_a=overwrite_a)
    l = np.diag(t, 1)  # noqa: E741
    return np.prod(sign(l[::2])) * la.det(z)

def pfaffian(A, method="P"):
    """Calculate Pfaffian of any antisymmetric matrix A using optimized C binding"""
    return 1 if np.real(cpf(A, method=method)) > 0.0 else -1