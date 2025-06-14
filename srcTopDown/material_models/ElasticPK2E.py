"""
File: Cartesian3d.py
Author: Tom de Geus copied to python
Date: 10.06.2025
Description: Nonlinear elastoplastic constitutive model following Simo and Tom de Geus

"""

import numpy as np
from .TensorFunc import * 


def Epseq(A):
    """
    Calculates the Von Mises equivalent strain from a 2nd-order strain tensor (or batch).

    Formula: `sqrt(2/3 * dev(A)_ij * dev(A)_ji)` or `sqrt(2/3) * Norm_deviatoric(A)`.

    Corresponds to the C++ `Epseq()` function.

    Args:
        A (np.ndarray): A 2nd-order strain tensor of shape `(..., 3, 3)`.

    Returns:
        np.ndarray: The Von Mises equivalent strain(s), shaped `(...)`.

    Raises:
        ValueError: If the input array does not end with a `(3, 3)` shape.
    """
    # Norm_deviatoric already handles batching and returns a (...,) array
    norm_dev = Norm_deviatoric(A)
    return np.sqrt(2.0 / 3.0) * norm_dev

# Note: The C++ `epseq(A, ret)` function which writes to an allocated output
# is generally not needed in Python. NumPy operations return new arrays. If in-place
# modification is strictly required for performance, one could consider a custom
# `out` argument, but it's less Pythonic.

def Sigeq(A):
    """
    Calculates the Von Mises equivalent stress from a 2nd-order stress tensor (or batch).

    Formula: `sqrt(3/2 * dev(A)_ij * dev(A)_ji)` or `sqrt(3/2) * Norm_deviatoric(A)`.

    Corresponds to the C++ `Sigeq()` function.

    Args:
        A (np.ndarray): A 2nd-order stress tensor of shape `(..., 3, 3)`.

    Returns:
        np.ndarray: The Von Mises equivalent stress(es), shaped `(...)`.

    Raises:
        ValueError: If the input array does not end with a `(3, 3)` shape.
    """
    # Norm_deviatoric already handles batching and returns a (...,) array
    norm_dev = Norm_deviatoric(A)
    return np.sqrt(1.5) * norm_dev

# Note: Similar to `epseq`, `sigeq(A, ret)` is not explicitly translated as
# in-place modification is less common and often less performant in Python/NumPy
# compared to returning new arrays.

def Strain(A):
    """
    Calculates the logarithmic strain tensor from the deformation gradient tensor F (or batch).

    The process is: `Strain = 0.5 * ln(B)`, where `B = F @ F.T` (Finger tensor).

    Corresponds to the C++ `Strain()` function.

    Args:
        A (np.ndarray): The deformation gradient tensor `F` of shape `(..., 3, 3)`.

    Returns:
        np.ndarray: The logarithmic strain tensor(s) of shape `(..., 3, 3)`.

    Raises:
        ValueError: If the input array does not end with a `(3, 3)` shape.
                    Also raises errors from `logs` if `B` is not symmetric positive-definite.
    """
    # Calculate the Finger tensor: B = F @ F.T
    B = A2_dot_A2T(A)
    # Calculate the matrix logarithm of B
    log_B = logs(B)
    # Scale by 0.5
    return 0.5 * log_B

# Note: Similar to previous functions, `strain(A, ret)` is not explicitly translated.
class ElasticPK2E:
    """
    Represents an array of material points with an elastic constitutive response
    within a finite strain framework (total lagrangian formulation).

    This class manages the material properties (Bulk and Shear moduli),
    the deformation gradient, stress, and tangent stiffness for a batch of material points.

    The class works with the PK2 stress and the right Green strain tensor.
    """

    # Constants for 3D tensors (fixed to 3x3 matrices or 3x3x3x3 tensors)
    _ndim = 3
    _stride_tensor2 = 9  # 3 * 3
    _stride_tensor4 = 81 # 3 * 3 * 3 * 3

    def __init__(self, K, G):
        """
        Constructs the elastic material system.

        Args:
            K (np.ndarray): Bulk modulus per item, shape `(...)`.
            G (np.ndarray): Shear modulus per item, shape `(...)`.

        Raises:
            ValueError: If `K` and `G` do not have matching shapes.
        """
        self.m_K = np.asarray(K)
        self.m_G = np.asarray(G)

        # Validate input shapes for K and G
        if self.m_K.shape != self.m_G.shape:
            raise ValueError("Shapes of K and G must match.")

        # Determine the batch shape from K (or G)
        self.m_shape = self.m_K.shape
        self.m_size = self.m_K.size # Total number of items in the batch

        # Define shapes for batched 2nd-order and 4th-order tensors
        self.m_shape_tensor2 = self.m_shape + (self._ndim, self._ndim)
        self.m_shape_tensor4 = self.m_shape + (self._ndim, self._ndim, self._ndim, self._ndim)

        # Initialize deformation gradient to identity for all items in the batch
        self.m_F = np.zeros(self.m_shape_tensor2, dtype=float)
        for idx in np.ndindex(self.m_shape):
            self.m_F[idx] = I2() # Set each 3x3 tensor to identity

        # Initialize stress and tangent containers as empty (will be filled by refresh)
        self.m_Sig = np.empty(self.m_shape_tensor2, dtype=float)
        self.m_PK2 = np.empty(self.m_shape_tensor2, dtype=float)
        self.m_C = np.empty(self.m_shape_tensor4, dtype=float)

        # Perform initial refresh to set stress and tangent based on initial F
        self.refresh()

    @property
    def K(self):
        """Bulk modulus per item."""
        return self.m_K

    @property
    def G(self):
        """Shear modulus per item."""
        return self.m_G

    @property
    def F(self):
        """Deformation gradient tensor per item."""
        return self.m_F

    @F.setter
    def F(self, arg):
        """
        Sets the deformation gradient tensors and triggers a refresh.

        Args:
            arg (np.ndarray): New deformation gradient tensor(s) with shape `(..., 3, 3)`.

        Raises:
            ValueError: If the input shape does not match the expected shape.
        """
        new_F = np.asarray(arg)
        if new_F.shape != self.m_shape_tensor2:
            raise ValueError(f"Input F shape {new_F.shape} does not match expected shape {self.m_shape_tensor2}.")
        self.m_F[:] = new_F # Assign in-place to avoid creating new object if not needed
        self.refresh() # Automatically refresh stress and tangent

    @property
    def Sig(self):
        """Cauchy stress tensor per item."""
        return self.m_Sig

    @property
    def PK2(self):
        """PK2 stress tensor per item."""
        return self.m_PK2

    @property
    def C(self):
        """Tangent tensor per item."""
        return self.m_C

    def refresh(self, compute_tangent=True):
        """
        Recomputes stress and optionally tangent stiffness from the current deformation gradient.

        This function should be called manually if elements of `self.F` were modified
        in-place without using the `self.F = new_F` setter.

        Args:
            compute_tangent (bool): If True, also computes and updates the tangent tensor `self.m_C`.
                                    Defaults to True.
        """

        # Extract relevant batched arrays
        F = self.m_F
        K_mod = self.m_K[..., None, None, None, None] # Broadcast K for 4th-order tensor operations
        G_mod = self.m_G[..., None, None, None, None] # Broadcast G for 4th-order tensor operations

        I = np.eye(3)  # (3,3)
        # Create 4th order identity tensors (3x3x3x3)
        I4_sym = 0.5 * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))  # (3,3,3,3)
        I4_vol = np.einsum('ij,kl->ijkl', I, I)  # (3,3,3,3)
    
        # Add axes to I4 tensors for broadcasting with batch dims (...,3,3,3,3)
        I4_sym = I4_sym[None, None, :, :, :, :]
        I4_vol = I4_vol[None, None, :, :, :, :]

        CSE = K_mod * I4_vol + 2 * G_mod * (I4_sym - (1.0/3.0) * I4_vol)
        
        # right cauchy-green tensor: Ce = F.T @ F
        Ce = A2T_dot_A2(F) # (nelem, nIP, 3, 3)

        # Green strain tensor: E = 0.5 (Ce - I)
        Ee = 0.5*(Ce - I2()[None, None, :, :]) # (nelem, nIP, 3, 3)

        self.m_PK2 = A4_dot_B2(CSE, Ee)

        # Compute Jacobian determinant J for each element and integration point
        J = np.linalg.det(F)  # shape (nelem, nIP)
        
        # Compute intermediate product: F @ S
        temp = np.einsum('abij,abjk->abik', F, self.PK2) # shape (nelem, nIP, 3, 3)

        # Compute Cauchy stress: (1/J) * temp @ F^T
        # F^T:
        F_T = np.swapaxes(F, -1, -2)  # transpose last two dims: shape (nelem, nIP, 3, 3)

        # Final:
        self.m_Sig = np.einsum('abij,abjk->abik', temp, F_T) / J[..., None, None]

        # Compute Cauchy stress, in original coordinate frame
        # self.m_Sig[:] = from_eigs(vec, Sig_val) # Reconstructs (..., 3, 3) tensor

        if not compute_tangent:
            return

        # material stiffness matrix
        # Cmat = dP/dF = F(i,M) * dS(M,J)/dE(N,P) dE(N,P/dF(k,L) 
        Kmat = np.einsum('ab i M, ab M J N L, ab k N -> ab i J k L', F, CSE, F)
        # geometrical stiffness matrix
        delta = I2()[None, None, :, :]  # (1,1,3,3)

        # Now build Kgeo shape: (nelem, nIP, 3, 3, 3, 3)
        Kgeo = (delta[..., :, :, None, None] * self.PK2[..., None, None, :, :]).transpose(0, 1, 4, 2, 3, 5)

        # Combine tangents: C = Kgeo + Kmat
        self.m_C[:] = Kgeo + Kmat 

class LinearHardeningPK2E:
    """
    Python translation of the C++ LinearHardening class,
    adapted for 2nd Piola-Kirchhoff stress (S) and Green-Lagrange strain (E).
    """

    def __init__(self, K, G, tauy0, H):
        """
        Construct system.
        :param K: Bulk modulus per item (NumPy array).
        :param G: Shear modulus per item (NumPy array).
        :param tauy0: Initial yield stress per item (NumPy array).
        :param H: Hardening modulus per item (NumPy array).
        """
        # Simulate base class members
        self.m_ndim = 3 # Hardcoded for Cartesian3d
        self.m_stride_tensor2 = self.m_ndim * self.m_ndim # 9
        self.m_stride_tensor4 = self.m_ndim * self.m_ndim * self.m_ndim * self.m_ndim # 81

        # Assert input shapes and set main shape
        assert K.ndim == 1, "K must be a 1D array."
        assert K.shape == G.shape, "K and G must have the same shape."
        assert K.shape == tauy0.shape, "K and tauy0 must have the same shape."
        assert K.shape == H.shape, "K and H must have the same shape."

        self.m_shape = K.shape # e.g., (size,)
        self.m_size = self.m_shape[0] # Number of material points

        # Calculate shapes for tensors
        self.m_shape_tensor2 = self.m_shape + (self.m_ndim, self.m_ndim) # e.g., (size, 3, 3)
        self.m_shape_tensor4 = self.m_shape + (self.m_ndim, self.m_ndim, self.m_ndim, self.m_ndim) # e.g., (size, 3, 3, 3, 3)

        # Initialize data members (NumPy arrays)
        self.m_K = K.copy()
        self.m_G = G.copy()
        self.m_tauy0 = tauy0.copy()
        self.m_H = H.copy()

        self.m_epsp = np.zeros(self.m_shape, dtype=float)       # Scalar equivalent plastic strain
        self.m_epsp_t = self.m_epsp.copy()                      # Previous equivalent plastic strain

        self.m_F = np.tile(np.eye(self.m_ndim), (self.m_size, 1, 1)) # Deformation gradient
        self.m_F_t = self.m_F.copy()                                # Previous deformation gradient

        # New members for S-E formulation
        self.m_E_p = np.zeros(self.m_shape_tensor2, dtype=float) # Tensorial plastic Green strain
        self.m_E_p_t = self.m_E_p.copy()                           # Previous plastic Green strain

        self.m_S = np.zeros(self.m_shape_tensor2, dtype=float)   # 2nd Piola-Kirchhoff stress
        self.m_S_t = self.m_S.copy()                             # Previous 2nd Piola-Kirchhoff stress

        # The tangent stiffness tensor dS/dE
        self.m_C = np.empty(self.m_shape_tensor4, dtype=float)

        self.refresh() # Initial refresh after construction

    # --- Getters (remain largely the same for scalar properties) ---
    def K(self):
        return self.m_K

    def G(self):
        return self.m_G

    def tauy0(self):
        return self.m_tauy0

    def H(self):
        return self.m_H

    def S(self):
        """2nd Piola-Kirchhoff stress per item."""
        return self.m_S

    def Ep(self):
        """Tensorial plastic Green strain per item."""
        return self.m_E_p

    def C(self):
        """Tangent stiffness tensor dS/dE per item."""
        return self.m_C

    # --- Setters ---
    def set_F(self, arg, compute_tangent=True):
        """
        Set deformation gradient tensors.
        Internally, this calls refresh() to update stress.
        :param arg: Deformation gradient tensor per item [shape(), 3, 3] (NumPy array).
        :param compute_tangent: Compute tangent (boolean).
        """
        assert arg.shape == self.m_shape_tensor2, "Input 'arg' shape does not match m_shape_tensor2."
        # Update previous step's F, S, Ep before setting new F and refreshing
        self.m_F_t[:] = self.m_F[:]
        self.m_S_t[:] = self.m_S[:]
        self.m_E_p_t[:] = self.m_E_p[:]
        self.m_epsp_t[:] = self.m_epsp[:]

        self.m_F[:] = arg # Assign the new deformation gradient
        self.refresh(compute_tangent)

    def refresh(self, compute_tangent=True):
        """
        Recompute stress from deformation gradient tensor using S-E formulation.
        """
        # Global constants (isotropic tensor definitions)
        II = GMatTensorCartesian3d.II()
        I4s = GMatTensorCartesian3d.I4s()

        # Loop over each material point
        for i in range(self.m_size):
            # Extract scalar properties for the current item
            K = self.m_K[i]
            G = self.m_G[i]
            tauy0 = self.m_tauy0[i]
            H = self.m_H[i]
            epsp_t = self.m_epsp_t[i]

            # Extract relevant batched arrays
            F = self.m_F
            K_mod = self.m_K[..., None, None, None, None] # Broadcast K for 4th-order tensor operations
            G_mod = self.m_G[..., None, None, None, None] # Broadcast G for 4th-order tensor operations

            I = np.eye(3)  # (3,3)
            # Create 4th order identity tensors (3x3x3x3)
            I4_sym = 0.5 * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))  # (3,3,3,3)
            I4_vol = np.einsum('ij,kl->ijkl', I, I)  # (3,3,3,3)
        
            # Add axes to I4 tensors for broadcasting with batch dims (...,3,3,3,3)
            I4_sym = I4_sym[None, None, :, :, :, :]
            I4_vol = I4_vol[None, None, :, :, :, :]

            CSE = K_mod * I4_vol + 2 * G_mod * (I4_sym - (1.0/3.0) * I4_vol)
            
            # right cauchy-green tensor: Ce = F.T @ F
            Ce = A2T_dot_A2(F) # (nelem, nIP, 3, 3)

            # Green strain tensor: E = 0.5 (Ce - I)
            Ee = 0.5*(Ce - I2()[None, None, :, :]) # (nelem, nIP, 3, 3)

            PK2_trial = A4_dot_B2(CSE, Ee)

            # compute trace along the last two axes
            trace_PK2 = np.trace(PK2_trial, axis1=-2, axis2=-1)  # shape (n_elem, n_gauss)

            # reshape trace to broadcast over the last two axes
            trace_PK2_expanded = trace_PK2[:, :, None, None]

            # 5. Calculate deviatoric trial stress and equivalent stress
            PK2_trial_dev = PK2_trial - (1/3) * trace_PK2_expanded * I2()[None, None, :, :]

            # compute eq. von-mises stress using PK2
            PK2_trial_eq = np.sqrt(1.5 * np.sum(PK2_trial_dev**2, axis=(-2, -1)))

            # 6. Evaluate the yield surface
            phi = PK2_trial_eq - (tauy0 + H * epsp_t)

            dgamma = 0.0 # Plastic multiplier
            PK2_current = np.copy(PK2_trial) # Initialize S_current with trial stress
            E_p_current = np.copy(E_p_t) # Initialize E_p_current with previous plastic strain
            epsp_current = epsp_t        # Initialize equivalent plastic strain

            # 7. Return Map (Radial Return)
            if phi > 0:
                # Calculate plastic multiplier
                # Denominator of delta lambda
                denominator = (3.0 * G + H)
                if denominator == 0: # Avoid division by zero
                    dgamma = 0.0
                else:
                    dgamma = phi / denominator

                # Normalized flow direction (deviatoric trial stress direction)
                if taueq_trial != 0:
                    N_flow = PK2_trial_dev / PK2_trial_eq
                else:
                    N_flow = np.zeros_like(S_trial_dev) # No flow direction if no stress

                # Update stress: S_current = S_trial - dgamma * (3G) * N_flow
                PK2_current = S_trial - dgamma * (3.0 * G) * N_flow

                # Update tensorial plastic strain: E_p_current = E_p_t + dgamma * N_flow
                # shouldn't that be: ept - dgamma * nflow?
                E_p_current = E_p_t + dgamma * N_flow

                # Update equivalent plastic strain: epsp_current = epsp_t + dgamma
                epsp_current = epsp_t + dgamma

            # Update class members for the current item
            self.m_S[i] = PK2_current
            self.m_E_p[i] = E_p_current
            self.m_epsp[i] = epsp_current

            # Compute tangent stiffness (dS/dE)
            if not compute_tangent:
                return # Exit the function entirely if tangent is not needed for any element.

            # Linearization of the constitutive response
            # use the correct structure here !! (nelem, nIP, 3,3,3,3)
            tangent_material = np.empty((self.m_ndim, self.m_ndim, self.m_ndim, self.m_ndim), dtype=float)

            if phi <= 0: # Elastic loading or unloading
                tangent_material[:] = CSE # Elastic tangent
            else: # Plastic loading
                # This part is an adaptation of the original tangent logic to S-E space.
                # It assumes the functional form of the tangent in the yield surface normal
                # and deviatoric components remains similar.
                a0 = 0.0
                if dgamma != 0.0 and taueq_trial != 0.0:
                    a0 = dgamma * G / taueq_trial # Original used G, not 3G

                a1 = G / (H + 3.0 * G) if (H + 3.0 * G) != 0 else 0.0

                NN_flow_dyadic = np.einsum('ij,kl->ijkl', N_flow, N_flow)

                # Adapted elastic-plastic tangent based on original structure
                tangent_material[:] = ( (K - (2.0 / 3.0) * G) + 2.0 * a0 * G ) * II + \
                                      (1.0 - 3.0 * a0) * G * I4s + \
                                      2.0 * G * (a0 - a1) * NN_flow_dyadic

            # Final tangent (only material tangent in S-E, no geometric part included)
            self.m_C[i] = tangent_material