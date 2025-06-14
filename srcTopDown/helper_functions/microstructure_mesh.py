"""

UNDER CONSTRUCTION - NOT USABLE YET

"""

import numpy as np
import random

def randomize_microstructure(elements, mesh, fractions, chunk  , K, G, tauy0, H):
    # Validate input shapes
    if len(K) > len(fractions):
        raise ValueError("K cannot be larger than the length of fractions.")
    if len(G) > len(fractions):
        raise ValueError("G cannot be larger than the length of fractions.")
    if len(tauy0) > len(fractions):
        raise ValueError("tauy0 cannot be larger than the length of fractions.")
    if len(H) > len(fractions):
        raise ValueError("H cannot be larger than the length of fractions.") 


    return K, G, tauy0, H

class Microstructure:
    def __init__(self, K, G):
        """
        Constructs the Microstructure.

        Args:
            elements (np.ndarray):
            mesh (np.nparray):
            fractions (np.array): percentage of different phases
            K (np.array): Bulk modulus of different phases
            G (np.array): Shear modulus of different phases
            tauy0 (np.array): Bulk modulus of different phases
            H (np.array): Shear modulus of different phases
        Raises:
            ValueError: If sum(fractions) != 1.
            ValueError: If array length of `K`, `G`, tauy0 or H is higher than len(fractions).
        Returns:
            K in shape (nelem, nIP, 1).
            G in shape (nelem, nIP, 1).
            tauy0 in shape (nelem, nIP, 1).
            H in shape (nelem, nIP, 1).
        """
        self.m_K = K
        self.m_G = G
        self.m_tauy0 = tauy0
        self.m_H = H

        # Validate input shapes for K and G
        if len(K) > len(fractions):
            raise ValueError("K cannot be larger than the length of fractions.")
        if len(G) > len(fractions):
            raise ValueError("G cannot be larger than the length of fractions.")
        if len(tauy0) > len(fractions):
            raise ValueError("tauy0 cannot be larger than the length of fractions.")
        if len(H) > len(fractions):
            raise ValueError("H cannot be larger than the length of fractions.")                        

        def random():

            return