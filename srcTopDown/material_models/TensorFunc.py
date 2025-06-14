import numpy as np

def I2():
    """
    Creates a 3x3 second-order identity tensor (matrix).

    Returns:
        np.ndarray: A 3x3 identity matrix.
    """
    return np.eye(3)

def II():
    """
    Creates a fourth-order tensor with components I_ijkl = delta_ij * delta_kl.
    This corresponds to the dyadic product of two second-order identity tensors (I otimes I).

    Returns:
        np.ndarray: A 3x3x3x3 fourth-order tensor.
    """
    I = np.eye(3) # Get the 2nd-order identity tensor

    # Using np.einsum:
    # 'ij,kl->ijkl' means: I_ij * I_kl -> result_ijkl
    result_II = np.einsum('ij,kl->ijkl', I, I)

    return result_II    

def I4():
    """
    Creates the fourth-order identity tensor with components I4_ijkl = delta_il * delta_jk.
    This is often denoted as the 'symmetric identity tensor' or 'permutation tensor'
    in some contexts, though its common name is just the fourth-order identity.

    Returns:
        np.ndarray: A 3x3x3x3 fourth-order tensor.
    """
    I = np.eye(3) # Get the 2nd-order identity tensor

    # Using np.einsum:
    # 'il,jk->ijkl' means: I_il * I_jk -> result_ijkl
    result_I4 = np.einsum('il,jk->ijkl', I, I)

    return result_I4    

def I4rt():
    """
    Creates the standard fourth-order identity tensor with components I4rt_ijkl = delta_ik * delta_jl.

    Returns:
        np.ndarray: A 3x3x3x3 fourth-order tensor.
    """
    I = np.eye(3) # Get the 2nd-order identity tensor (delta_ij)

    # Using np.einsum:
    # 'ik,jl->ijkl' means: I_ik * I_jl -> result_ijkl
    # This directly implements the desired Kronecker delta product.
    result_I4rt = np.einsum('ik,jl->ijkl', I, I)

    return result_I4rt    

import numpy as np

# We'll re-use the I4 and I4rt functions you already have in Python:
# def I4(): ... (calculates delta_il * delta_jk)
# def I4rt(): ... (calculates delta_ik * delta_jl)

# Using the definitions from previous conversions:
def I4_delta_il_delta_jk():
    I = np.eye(3)
    return np.einsum('il,jk->ijkl', I, I)

def I4rt_delta_ik_delta_jl():
    I = np.eye(3)
    return np.einsum('ik,jl->ijkl', I, I)

def I4s():
    """
    Creates the fourth-order symmetric identity tensor, I4s_ijkl = 0.5 * (delta_il * delta_jk + delta_ik * delta_jl).

    Returns:
        np.ndarray: A 3x3x3x3 fourth-order tensor.
    """
    tensor_I4 = I4_delta_il_delta_jk()     # Corresponds to C++'s I4(ret)
    tensor_I4rt = I4rt_delta_ik_delta_jl() # Corresponds to C++'s I4rt(&i4rt[0])

    # Summing the two tensors and multiplying by 0.5
    symmetric_tensor = 0.5 * (tensor_I4 + tensor_I4rt)

    return symmetric_tensor    

def I4d():
    """
    Creates the fourth-order deviatoric identity tensor, I4d = I4s - (1/3) * (I otimes I).

    Returns:
        np.ndarray: A 3x3x3x3 fourth-order tensor.
    """
    tensor_I4s = I4s() # Get I4s
    tensor_II = II()   # Get I otimes I

    # Calculate (1/3) * (I otimes I)
    third_II = (1.0 / 3.0) * tensor_II

    # Subtract to get the deviatoric tensor
    deviatoric_tensor = tensor_I4s - third_II

    return deviatoric_tensor    

def Trace(A):
    """
    Computes the trace of a 2nd-order tensor (matrix).

    Args:
        A (np.ndarray): A 2nd-order tensor (3x3 matrix).

    Returns:
        float: The trace of the tensor.
    """
    # Ensure A is a NumPy array for np.trace to work seamlessly
    A = np.asarray(A)
    # Check shape if necessary, e.g., if A.shape != (3, 3)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input 'A' must be a square 2nd-order tensor (matrix).")

    return np.trace(A)    

def Hydrostatic(A):
    """
    Computes the hydrostatic part of a 2nd-order tensor.

    Args:
        A (np.ndarray): A 2nd-order tensor (3x3 matrix).

    Returns:
        float: The hydrostatic part of the tensor.
    """
    return Trace(A) / 3.0    

def Det(A):
    """
    Computes the determinant of a 2nd-order tensor (3x3 matrix).

    Args:
        A (np.ndarray): A 3x3 matrix.

    Returns:
        float: The determinant of the matrix.
    """
    # Ensure A is a NumPy array
    A = np.asarray(A)
    # Check shape to ensure it's a 3x3 matrix for this specific function context
    if A.shape != (3, 3):
        raise ValueError("Input 'A' must be a 3x3 matrix for this function.")

    return np.linalg.det(A)    

def sym(A):
    """
    Computes the symmetric part of a 2nd-order tensor (matrix).

    Args:
        A (np.ndarray): A 3x3 matrix.

    Returns:
        np.ndarray: The symmetric part of the matrix.
    """
    A = np.asarray(A)
    if A.shape != (3, 3):
        raise ValueError("Input 'A' must be a 3x3 matrix for this function.")

    return 0.5 * (A + A.T) # A.T computes the transpose    


def Inv(A):
    """
    Computes the inverse of a 2nd-order tensor (3x3 matrix) and returns its determinant.

    Args:
        A (np.ndarray): A 3x3 matrix.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The inverse matrix.
            - float: The determinant of the original matrix.
    """
    A = np.asarray(A)
    if A.shape != (3, 3):
        raise ValueError("Input 'A' must be a 3x3 matrix for this function.")

    D = np.linalg.det(A)
    # Check for singularity
    if D == 0:
        raise ValueError("Matrix is singular; inverse does not exist.")

    inv_A = np.linalg.inv(A) # NumPy's built-in inverse function

    return inv_A, D    


def Hydrostatic_deviatoric(A):
    """
    Computes the hydrostatic part and the deviatoric part of a 2nd-order tensor (matrix).

    Args:
        A (np.ndarray): A 3x3 matrix.

    Returns:
        tuple: A tuple containing:
            - float: The hydrostatic part (mean stress/strain).
            - np.ndarray: The deviatoric part of the matrix.
    """
    A = np.asarray(A)
    if A.shape != (3, 3):
        raise ValueError("Input 'A' must be a 3x3 matrix for this function.")

    hydrostatic_part = Hydrostatic(A)
    identity_matrix = np.eye(3) # 3x3 identity matrix

    # Deviatoric part = A - m * I
    deviatoric_part = A - hydrostatic_part * identity_matrix

    return hydrostatic_part, deviatoric_part    


# --- Double Tensor Contraction of a Tensor's Deviator with Itself ---
def Deviatoric_ddot_deviatoric(A):
    """
    Computes the double tensor contraction of a tensor's deviator with itself: `dev(A) : dev(A)`.

    This is equivalent to the squared Frobenius norm of the deviatoric part.
    It's a common stress invariant used in plasticity models.

    Corresponds to the C++ `Deviatoric_ddot_deviatoric()` function.

    Args:
        A (np.ndarray): A 3x3 matrix.

    Returns:
        float: The scalar result of the double dot product.

    Raises:
        ValueError: If the input array is not a 3x3 matrix.
    """
    A = np.asarray(A)
    # Validate input shape for a 3x3 matrix.
    if A.shape != (3, 3):
        raise ValueError("Input 'A' must be a 3x3 matrix for this function.")

    # Get the deviatoric part of A using the previously defined function.
    _, dev_A = Hydrostatic_deviatoric(A)

    # The double dot product A:B (or A_ij * B_ij summed over i and j) is
    # equivalent to element-wise multiplication followed by summation.
    return np.sum(dev_A * dev_A)

# --- Norm of the Deviatoric Part ---

def Norm_deviatoric(A):
    """
    Computes the Frobenius norm of the deviatoric part of a 2nd-order tensor.

    This is the square root of `Deviatoric_ddot_deviatoric(A)`. It provides
    a measure of the magnitude of the shear components in the tensor.

    Corresponds to the C++ `Norm_deviatoric()` function.

    Args:
        A (np.ndarray): A 3x3 matrix.

    Returns:
        float: The scalar norm of the deviatoric part.

    Raises:
        ValueError: If the input array is not a 3x3 matrix.
    """
    A = np.asarray(A)
    # Validate input shape for a 3x3 matrix.
    if A.shape != (3, 3):
        raise ValueError("Input 'A' must be a 3x3 matrix for this function.")

    # Calls Deviatoric_ddot_deviatoric and takes the square root.
    return np.sqrt(Deviatoric_ddot_deviatoric(A))

# --- Double Tensor Contraction of Two 2nd-Order Tensors (General) ---

def A2_ddot_B2(A, B):
    """
    Computes the double tensor contraction of two general 2nd-order tensors A and B: `A : B = A_ij * B_ij`.

    This is also known as the Frobenius inner product of two matrices.

    Corresponds to the C++ `A2_ddot_B2()` function.

    Args:
        A (np.ndarray): The first 3x3 matrix.
        B (np.ndarray): The second 3x3 matrix.

    Returns:
        float: The scalar result of the double dot product.

    Raises:
        ValueError: If input arrays are not 3x3 matrices.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    # Validate input shapes.
    if A.shape != (3, 3) or B.shape != (3, 3):
        raise ValueError("Inputs A and B must be 3x3 matrices for this function.")

    # Performs element-wise multiplication (A * B) and then sums all elements.
    return np.sum(A * B)

# --- Double Tensor Contraction of Two Symmetric 2nd-Order Tensors ---

def A2s_ddot_B2s(A, B):
    """
    Computes the double tensor contraction of two symmetric 2nd-order tensors A and B.

    The formula used `A_00*B_00 + A_11*B_11 + A_22*B_22 + 2*(A_01*B_01 + A_02*B_02 + A_12*B_12)`
    is a common compact form for symmetric matrices, accounting for the repeated off-diagonal terms.

    Corresponds to the C++ `A2s_ddot_B2s()` function.

    Args:
        A (np.ndarray): The first symmetric 3x3 matrix.
        B (np.ndarray): The second symmetric 3x3 matrix.

    Returns:
        float: The scalar result of the double dot product.

    Raises:
        ValueError: If input arrays are not 3x3 matrices.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    # Validate input shapes.
    if A.shape != (3, 3) or B.shape != (3, 3):
        raise ValueError("Inputs A and B must be 3x3 matrices for this function.")

    # Explicitly implements the sum based on the symmetric properties, mirroring the C++ code.
    # For symmetric matrices, np.sum(A * B) would yield the same result.
    return (A[0,0] * B[0,0] + A[1,1] * B[1,1] + A[2,2] * B[2,2] +
            2 * A[0,1] * B[0,1] + 2 * A[0,2] * B[0,2] + 2 * A[1,2] * B[1,2])

def A2_dyadic_B2(A, B):
    """
    Computes the dyadic (outer) product of two 2nd-order tensors A and B,
    resulting in a 4th-order tensor.

    The result `C_ijkl = A_ij * B_kl`.

    Corresponds to the C++ `A2_dyadic_B2()` function.

    Args:
        A (np.ndarray): The first 3x3 matrix.
        B (np.ndarray): The second 3x3 matrix.

    Returns:
        np.ndarray: A 3x3x3x3 NumPy array representing the 4th-order dyadic product.

    Raises:
        ValueError: If input arrays are not 3x3 matrices.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != (3, 3) or B.shape != (3, 3):
        raise ValueError("Inputs A and B must be 3x3 matrices for this function.")

    # np.einsum is ideal for dyadic products (outer products).
    return np.einsum('ij,kl->ijkl', A, B)

# --- Single Contraction: 4th-order A dot 2nd-order B (A_ijkl * B_lm -> A_ijkm) ---

def A4_dot_B2(A, B):
    """
    Batched contraction of 4th-order tensor A with 2nd-order tensor B over the index 'l'.

    Args:
        A (np.ndarray): shape (..., 3, 3, 3, 3)
        B (np.ndarray): shape (..., 3, 3)
    
    Returns:
        np.ndarray: shape (..., 3, 3)
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape[-4:] != (3, 3, 3, 3) or B.shape[-2:] != (3, 3):
        raise ValueError("Input A must have last 4 dims (3,3,3,3) and B last 2 dims (3,3).")

    # Use einsum with ellipsis '...' to handle arbitrary batch dims,
    # Contract over 'l': A_ijkl * B_lm -> ret_ijkm
    # Then, to get shape (..., 3, 3), contract over the last two indices appropriately:
    # So we do '...ijkl,...lm->...ijkm', then contract last two dims (k,m) into 3x3 by swapping dims or summing if needed.
    # Actually, since you want (...,3,3) output, the operation is usually: ret_ij = A_ijkl * B_lm summed over l,m properly.

    # But your original contract gives output (...,3,3,3), so you might want to sum over k or m?
    # Check your mathematical operation!

    # If your intention is to get PK2_ij = A_ijkl * Ee_lk (usual for J2 plasticity), you want contraction over l and k:
    # einsum: 'ijkl,lk->ij'

    # So double check the math:

    # For your example, likely:
    # PK2_{ij} = A_{ijkl} * Ee_{kl}
    # So the operation is 'ijkl,kl->ij'

    # To support batches, it’s:

    return np.einsum('...ijkl,...kl->...ij', A, B)

# --- Matrix Multiplication: 2nd-order A dot 2nd-order B ---

def A2_dot_B2_matrix_mult(A, B):
    """
    Performs standard matrix multiplication (dot product) of two 2nd-order tensors A and B.

    The result `C_ik = A_ij * B_jk` (summation over `j`).
    This is equivalent to `np.matmul(A, B)`.

    Corresponds to the C++ `A2_dot_B2()` function.

    Args:
        A (np.ndarray): The first 3x3 matrix.
        B (np.ndarray): The second 3x3 matrix.

    Returns:
        np.ndarray: A 3x3 NumPy array representing the resulting matrix.

    Raises:
        ValueError: If input arrays are not 3x3 matrices.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != (3, 3) or B.shape != (3, 3):
        raise ValueError("Inputs A and B must be 3x3 matrices for this function.")

    # np.matmul or the @ operator are the standard for matrix multiplication in NumPy.
    return np.matmul(A, B)

# --- Product of 2nd-order A with its Transpose (A.T * A) ---

def A2T_dot_A2(A):
    """
    Computes the matrix product of a 2nd-order transposed tensor A.T with A.
    Supports batched inputs with shape (..., 3, 3).

    Args:
        A (np.ndarray): Array with last two dims (3,3), possibly batched.

    Returns:
        np.ndarray: Batched matrix product of shape (..., 3, 3),
                    where each is A[i].T @ A[i].

    Raises:
        ValueError: If the last two dimensions are not (3,3).
    """
    A = np.asarray(A)
    if A.shape[-2:] != (3, 3):
        raise ValueError("Input 'A' must have last two dims equal to (3,3).")

    # Perform batched matrix multiplication of A.T @ A
    # Swap the last two dims of A to get A^T
    A_T = np.swapaxes(A, -1, -2)
    return np.matmul(A_T, A)

# --- Product of 2nd-order A with its Transpose (A * A.T) ---

def A2_dot_A2T(A):
    """
    Computes the matrix product of a 2nd-order tensor A with its transpose A.T.

    The result `C_ik = A_ij * (A_T)_jk = A_ij * A_kj` (summation over `j`).
    This is equivalent to `np.matmul(A, A.T)`.

    Corresponds to the C++ `A2_dot_A2T()` function.

    Args:
        A (np.ndarray): A 3x3 matrix.

    Returns:
        np.ndarray: A 3x3 NumPy array representing the resulting matrix.

    Raises:
        ValueError: If the input array is not a 3x3 matrix.
    """
    A = np.asarray(A)
    if A.shape != (3, 3):
        raise ValueError("Input 'A' must be a 3x3 matrix for this function.")

    # np.matmul or the @ operator are the standard for matrix multiplication.
    return np.matmul(A, A.T)

# --- Double Contraction: 4th-order A double dot 2nd-order B ---

def A4_ddot_B2_double_contract(A, B):
    """
    Computes the double tensor contraction of a 4th-order tensor A with a 2nd-order tensor B.

    The operation is defined as `ret_ij = A_ijkl * B_lk` (summation over `k` and `l`).
    This means the result `ret` is a 2nd-order tensor.
    This operation is equivalent to A : B^T in some continuum mechanics notations.

    Corresponds to the C++ `A4_ddot_B2()` function.

    Args:
        A (np.ndarray): A 3x3x3x3 4th-order tensor.
        B (np.ndarray): A 3x3 2nd-order tensor.

    Returns:
        np.ndarray: A 3x3 NumPy array representing the resulting 2nd-order tensor.

    Raises:
        ValueError: If input arrays have incorrect shapes.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != (3, 3, 3, 3) or B.shape != (3, 3):
        raise ValueError("Input A must be a 3x3x3x3 tensor and B must be a 3x3 matrix.")

    # Using Einstein summation: 'ijkl,lk->ij' contracts on the 'k' and 'l' indices.
    return np.einsum('ijkl,lk->ij', A, B)

# --- Triple Double Contraction: 4th-order A : 4th-order B : 4th-order C ---

def A4_ddot_B4_ddot_C4(A, B, C):
    """
    Computes a triple double tensor contraction of three 4th-order tensors A, B, and C.

    The operation is defined by the index notation: `D_ijop = A_ijkl * B_lkmn * C_nmop`
    (summation over `k`, `l`, `m`, `n`).

    Corresponds to the C++ `A4_ddot_B4_ddot_C4()` function.

    Args:
        A (np.ndarray): The first 3x3x3x3 4th-order tensor.
        B (np.ndarray): The second 3x3x3x3 4th-order tensor.
        C (np.ndarray): The third 3x3x3x3 4th-order tensor.

    Returns:
        np.ndarray: A 3x3x3x3 NumPy array representing the resulting 4th-order tensor.

    Raises:
        ValueError: If input arrays have incorrect shapes.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    if A.shape != (3, 3, 3, 3) or B.shape != (3, 3, 3, 3) or C.shape != (3, 3, 3, 3):
        raise ValueError("Inputs A, B, C must be 3x3x3x3 tensors for this function.")

    # Einstein summation directly implements the multiple contractions.
    return np.einsum('ijkl,lkmn,nmop->ijop', A, B, C)

# --- Triple Matrix Product: A . B . C^T ---

def A2_dot_B2_dot_C2T(A, B, C):
    """
    Computes a triple matrix product of three 2nd-order tensors A, B, and the transpose of C.

    The operation is defined by the index notation: `D_il = A_ij * B_jk * C_lk`
    (summation over `j` and `k`). This is equivalent to `A @ B @ C.T`.

    Corresponds to the C++ `A2_dot_B2_dot_C2T()` function.

    Args:
        A (np.ndarray): The first 3x3 matrix.
        B (np.ndarray): The second 3x3 matrix.
        C (np.ndarray): The third 3x3 matrix.

    Returns:
        np.ndarray: A 3x3 NumPy array representing the resulting matrix.

    Raises:
        ValueError: If input arrays are not 3x3 matrices.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    if A.shape != (3, 3) or B.shape != (3, 3) or C.shape != (3, 3):
        raise ValueError("Inputs A, B, C must be 3x3 matrices for this function.")

    # Using np.matmul (or @ operator) for sequential matrix multiplications.
    return np.matmul(np.matmul(A, B), C.T)            

def eigs(A):
    """
    Computes the eigenvalues and eigenvectors of a 3x3 symmetric second-order tensor (matrix).
    This function handles batches of matrices.

    Corresponds to the C++ `eigs()` function, which internally uses
    a Jacobi algorithm for symmetric matrices. NumPy's `np.linalg.eigh` is
    the standard and highly optimized function for this in Python, specifically
    for symmetric (Hermitian) matrices, ensuring real eigenvalues and orthogonal eigenvectors.

    Args:
        A (np.ndarray): A symmetric matrix of shape `(..., 3, 3)`.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of eigenvalues with shape `(..., 3)` (sorted in ascending order).
            - np.ndarray: An array of eigenvectors with shape `(..., 3, 3)` where each
                          `eigenvectors[..., :, col_idx]` is an eigenvector.

    Raises:
        ValueError: If the input array does not end with a `(3, 3)` shape.
        UserWarning: If the input matrix (or any matrix in the batch) is not strictly symmetric.
    """
    A = np.asarray(A)
    if A.shape[-2:] != (3, 3):
        raise ValueError("Input 'A' must have shape (..., 3, 3).")
    # Check for symmetry. np.allclose is used due to floating-point comparisons.
    # This check applies to each matrix in the batch.
    if not np.all(np.allclose(A, A.swapaxes(-1, -2), rtol=1e-5, atol=1e-8)):
        print("Warning: Input matrix (or some in batch) is not strictly symmetric. np.linalg.eigh expects symmetric input.")

    # np.linalg.eigh returns eigenvalues in ascending order and eigenvectors as columns
    # for each matrix in the batch.
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    return eigenvalues, eigenvectors

# --- Reconstruct Symmetric Tensor from Eigenvalues and Eigenvectors (Batch Compatible) ---

def from_eigs(vec, val):
    """
    Reconstructs a symmetric 2nd-order tensor from its eigenvalues and eigenvectors.
    This function handles batches of eigenvalues and eigenvectors.

    The reconstruction uses the spectral decomposition formula: `A = sum(lambda_a * v_a otimes v_a)`,
    which is equivalent to the matrix multiplication `A = vec @ diag(val) @ vec.T`,
    where `vec` is the matrix of eigenvectors (columns) and `diag(val)` is a diagonal matrix
    of eigenvalues.

    Corresponds to the C++ `from_eigs()` function.

    Args:
        vec (np.ndarray): An array of eigenvectors with shape `(..., 3, 3)`, where
                          `vec[..., :, a]` is the `a`-th eigenvector.
        val (np.ndarray): An array of eigenvalues with shape `(..., 3)`.

    Returns:
        np.ndarray: A NumPy array of shape `(..., 3, 3)` representing the reconstructed symmetric tensor.

    Raises:
        ValueError: If input arrays have incorrect shapes.
    """
    vec = np.asarray(vec)
    val = np.asarray(val)
    if vec.shape[-2:] != (3, 3) or val.shape[-1] != 3:
        raise ValueError("Input 'vec' must have shape (..., 3, 3) and 'val' must have shape (..., 3).")
    if vec.shape[:-2] != val.shape[:-1]:
        raise ValueError("Batch dimensions of 'vec' and 'val' must match.")

    # Create a diagonal matrix from eigenvalues, preserving batch dimensions.
    # `val[..., None, :]` creates a (..., 1, 3) array
    # `val[..., :, None]` creates a (..., 3, 1) array
    # `np.diag` needs a 1D array per diagonal, so the batch dimension is handled by stacking.
    # The most robust way is to make `lambda_diag` have shape (..., 3, 3)
    lambda_diag = np.zeros(vec.shape[:-2] + (3, 3), dtype=vec.dtype)
    batch_indices = tuple(np.indices(val.shape[:-1]))
    lambda_diag[batch_indices + (np.arange(3), np.arange(3))] = val

    # Perform the spectral decomposition: A = V @ Lambda @ V.T
    # `vec.swapaxes(-1, -2)` is used for batch-compatible transpose.
    reconstructed_tensor = np.matmul(np.matmul(vec, lambda_diag), vec.swapaxes(-1, -2))
    return reconstructed_tensor

# --- Matrix Logarithm (based on eigenvalues, Batch Compatible) ---

def logs(A):
    """
    Computes the matrix logarithm of a 2nd-order tensor (matrix) or a batch of them,
    based on their eigenvalues.

    The operation involves:
    1. Finding the eigenvalues and eigenvectors of A.
    2. Taking the natural logarithm of each eigenvalue.
    3. Reconstructing the tensor using the new (log-transformed) eigenvalues and original eigenvectors.
    This approach is suitable for symmetric positive-definite matrices where the logarithm
    of eigenvalues is well-defined.

    Corresponds to the C++ `logs()` function (both `pointer::logs` and global `logs`).

    Args:
        A (np.ndarray): A symmetric positive-definite matrix of shape `(..., 3, 3)`.

    Returns:
        np.ndarray: A NumPy array of shape `(..., 3, 3)` representing the matrix logarithm.

    Raises:
        ValueError: If the input array has incorrect shape or contains non-positive eigenvalues.
        UserWarning: If the input matrix (or any in batch) is not strictly symmetric.
    """
    A = np.asarray(A)
    if A.shape[-2:] != (3, 3):
        raise ValueError("Input 'A' must have shape (..., 3, 3).")

    # Get eigenvalues and eigenvectors. Assumes A is symmetric.
    # The `eigs` function will handle symmetry warning.
    eigenvalues, eigenvectors = eigs(A)

    # Ensure all eigenvalues are positive before taking logarithm.
    if np.any(eigenvalues <= 0):
        raise ValueError("Matrix (or some in batch) has non-positive eigenvalues; cannot compute real logarithm.")

    # Take the natural logarithm of each eigenvalue.
    log_eigenvalues = np.log(eigenvalues)

    # Reconstruct the tensor using the new (log-transformed) eigenvalues.
    log_tensor = from_eigs(eigenvectors, log_eigenvalues)
    return log_tensor

# --- Random 2nd-Order Tensor (for testing) ---

def Random2(shape=None):
    """
    Generates a random 2nd-order tensor (matrix) or a batch of them.

    If `shape` is None, returns a 3x3 matrix.
    If `shape` is provided as a tuple, returns a `shape + (3, 3)` array.

    Corresponds to the C++ `Random2()` function.

    Args:
        shape (tuple, optional): The leading batch dimensions for the array of tensors.
                                 Defaults to None (returns a single 3x3 matrix).

    Returns:
        np.ndarray: A NumPy array with random elements, shaped `(..., 3, 3)`.
    """
    if shape is None:
        return np.random.randn(3, 3)
    else:
        return np.random.randn(*shape, 3, 3)

# --- Random 4th-Order Tensor (for testing) ---

def Random4(shape=None):
    """
    Generates a random 4th-order tensor or a batch of them.

    If `shape` is None, returns a 3x3x3x3 tensor.
    If `shape` is provided as a tuple, returns a `shape + (3, 3, 3, 3)` array.

    Corresponds to the C++ `Random4()` function.

    Args:
        shape (tuple, optional): The leading batch dimensions for the array of tensors.
                                 Defaults to None (returns a single 3x3x3x3 tensor).

    Returns:
        np.ndarray: A NumPy array with random elements, shaped `(..., 3, 3, 3, 3)`.
    """
    if shape is None:
        return np.random.randn(3, 3, 3, 3)
    else:
        return np.random.randn(*shape, 3, 3, 3, 3)

# --- 2nd-Order Null Tensor (all zeros) ---

def O2(shape=None):
    """
    Generates a 2nd-order null tensor (all components equal to zero) or a batch of them.

    If `shape` is None, returns a 3x3 matrix.
    If `shape` is provided as a tuple, returns a `shape + (3, 3)` array.

    Corresponds to the C++ `O2()` function.

    Args:
        shape (tuple, optional): The leading batch dimensions for the array of tensors.
                                 Defaults to None (returns a single 3x3 matrix).

    Returns:
        np.ndarray: A NumPy array filled with zeros, shaped `(..., 3, 3)`.
    """
    if shape is None:
        return np.zeros((3, 3))
    else:
        return np.zeros(shape + (3, 3))

# --- 4th-Order Null Tensor (all zeros) ---

def O4(shape=None):
    """
    Generates a 4th-order null tensor (all components equal to zero) or a batch of them.

    If `shape` is None, returns a 3x3x3x3 tensor.
    If `shape` is provided as a tuple, returns a `shape + (3, 3, 3, 3)` array.

    Corresponds to the C++ `O4()` function.

    Args:
        shape (tuple, optional): The leading batch dimensions for the array of tensors.
                                 Defaults to None (returns a single 3x3x3x3 tensor).

    Returns:
        np.ndarray: A NumPy array filled with zeros, shaped `(..., 3, 3, 3, 3)`.
    """
    if shape is None:
        return np.zeros((3, 3, 3, 3))
    else:
        return np.zeros(shape + (3, 3, 3, 3))

# --- Functions for array/shape metadata (equivalent to C++ underlying_size/shape) ---

def underlying_size(A):
    """
    Returns the product of the leading (batch) dimensions of a tensor array.
    This corresponds to the 'm_size' in the C++ `Array` class.

    Args:
        A (np.ndarray): A NumPy array with any number of leading (batch) dimensions
                        followed by a tensor's dimensions (e.g., (..., 3, 3)).

    Returns:
        int: The total number of tensors in the array.
    """
    # Assuming the last 2 or 4 dimensions are the tensor's own dimensions.
    # This function needs to determine the tensor's rank to correctly identify batch dimensions.
    # A more robust implementation might require passing the tensor's rank.
    # For now, let's assume it's either a 2nd-order or 4th-order tensor at the end.
    if A.ndim >= 4 and A.shape[-4:] == (3, 3, 3, 3):
        # It's an array of 4th-order tensors
        return np.prod(A.shape[:-4])
    elif A.ndim >= 2 and A.shape[-2:] == (3, 3):
        # It's an array of 2nd-order tensors
        return np.prod(A.shape[:-2])
    else:
        # It's a scalar array or something else
        return np.prod(A.shape)

def underlying_shape(A):
    """
    Returns the leading (batch) dimensions of a tensor array.
    This corresponds to the 'm_shape' in the C++ `Array` class.

    Args:
        A (np.ndarray): A NumPy array with any number of leading (batch) dimensions
                        followed by a tensor's dimensions (e.g., (..., 3, 3)).

    Returns:
        tuple: The tuple of leading (batch) dimensions.
    """
    # Similar to underlying_size, needs to infer tensor rank.
    if A.ndim >= 4 and A.shape[-4:] == (3, 3, 3, 3):
        return A.shape[:-4]
    elif A.ndim >= 2 and A.shape[-2:] == (3, 3):
        return A.shape[:-2]
    else:
        return A.shape    