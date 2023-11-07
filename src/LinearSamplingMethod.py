import numpy as np 
import numpy.typing as npt
from scipy.special import j0, y0
from scipy.spatial.distance import cdist
from numpy.linalg import svd

def LSM_scalar( FF: npt.NDArray[np.complex128], k: float,  x: float, y: float, alpha: float = 0):
    pass
    

def LSM_array( FF: npt.NDArray[np.complex128], 
                k: npt.NDArray[np.float64],
              r_R: npt.NDArray[np.float64],
                Z: npt.NDArray[np.float64],
                alpha: float = 0) -> npt.NDArray[np.float64]:
    
    kr = np.multiply.outer(k, cdist( r_R, Z, "euclidean" ))
    b = 1j/4*( j0(kr) + 1j*y0(kr) )

    u, s, _ = svd(FF,full_matrices=False)
    g2 = sum( ( (s[n]/(s[n]**2+a))*np.abs(np.vdot(u[:,n],b)) )**2 for n in range(len(u)) )
    return 1/g2
