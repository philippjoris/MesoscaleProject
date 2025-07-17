import argparse
import sys

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import GooseFEM
import numpy as np
# mesh
# ----

# define mesh
print("running a GooseFEM static example...")
mesh = GooseFEM.Mesh.Quad4.Regular(10, 10)

# mesh dimensions
nelem = mesh.nelem
nne = mesh.nne
ndim = mesh.ndim

# mesh definition, displacement, external forces
coor = mesh.coor
conn = mesh.conn
dofs = mesh.dofs
disp = np.zeros_like(coor)
fext = np.zeros_like(coor)

# list of prescribed DOFs
iip = np.concatenate(
    (
        # dofs[mesh.nodesRightEdge, 0],
        # dofs[mesh.nodesTopEdge, 1],
        dofs[mesh.nodesLeftEdge, 0],
        dofs[mesh.nodesBottomEdge, 1],
    )
)

# simulation variables
# --------------------

# vector definition
vector = GooseFEM.VectorPartitioned(conn, dofs, iip)

# allocate system matrix
K = GooseFEM.MatrixPartitioned(conn, dofs, iip)
Solver = GooseFEM.MatrixPartitionedSolver()

# element definition
elem = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor))
nip = elem.nip

# material definition
# -------------------
mat = GMat.Elastic2d(K=np.ones([nelem, nip]), G=np.ones([nelem, nip]))
# simulation variables
# --------------------
ue = vector.AsElement(disp)
coore = vector.AsElement(coor)
# this function calculates the gradient of the displacement field so I add the coordinates to the displacements
elem.gradN_vector((coore + ue), mat.F)
mat.F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
mat.refresh()

# internal force of the right hand side per element and assembly
fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
fint = vector.AssembleNode(fe)

# initial element tangential stiffness matrix incorporating the geometrical and material stiffness matrix
Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
K.assemble(Ke)

# initial residual
fres = fext - fint

# solve
# -----
ninc = 501
max_iter = 50
tangent = True

epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

curr_increment = np.zeros_like(vector.AsDofs_u(disp))
total_increment = np.zeros_like(vector.AsDofs_u(disp))

initial_guess = np.zeros_like(vector.AsDofs_u(disp))

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    # store old displacement
    xp = vector.AsDofs_p(disp).copy()

    # introduce new displacement increment
    fext[mesh.nodesRightEdge, 0] += (+2.0/ninc)
    fext[mesh.nodesTopEdge, 1] += (-2.0/ninc)
    disp[mesh.nodesLeftEdge, 0] = 0.0  # not strictly needed: default == 0
    disp[mesh.nodesBottomEdge, 1] = 0.0  # not strictly needed: default == 0

    # new displacement increment
    delta_xp = vector.AsDofs_p(disp) - xp
    # convergence flag
    converged = False
    # impose initial guess on unknown displacements
    disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + initial_guess, vector.AsDofs_p(disp))

    curr_increment = total_increment.copy()
    total_increment = initial_guess
    # vector.copy_p(fint, fext)
    for iter in range(max_iter): 
        # update element wise displacments
        ue = vector.AsElement(disp) 

        # update deformation gradient F
        elem.gradN_vector((coore + ue), mat.F)
        mat.F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
        mat.refresh()  
        
        # F.shape = (elem, IP, 3, 3)
        J = np.linalg.det(mat.F)  # J.shape = (elem, IP)
        F_inv = np.linalg.inv(mat.F)
        F_inv_T = np.transpose(F_inv, axes=(0, 1, 3, 2))
        tau = J[..., None, None] * mat.Sig
        P = np.matmul(tau, F_inv_T)
        P_T = np.transpose(P, axes=(0, 1, 3, 2))
  
        # update internal forces and assemble
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        fint = vector.AssembleNode(fe)

        # residual
        fres = vector.AsDofs_u(fext)-vector.AsDofs_u(fint)
        
        res_norm = np.linalg.norm(fres) 
        if res_norm < 1e-06:
            print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
            converged = True
            break


        # update stiffness matrix
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
        K.assemble(Ke)

        # solve
        if iter == 0:
            Solver.solve_u(K, fres, delta_xp, curr_increment)
        else:
            Solver.solve_u(K, fres, np.zeros_like(xp), curr_increment)

        # add newly found delta_u to total increment
        total_increment += curr_increment
        # update displacement vector
        disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + curr_increment, vector.AsDofs_p(disp))
    if converged:
        initial_guess = total_increment
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")

print(fint)
# post-process
# ------------
# strain
vector.asElement(disp, ue)
elem.symGradN_vector(ue, mat.F)
# mat.refresh()

# internal force
elem.int_gradN_dot_tensor2_dV(mat.Sig, fe)
vector.assembleNode(fe, fint)

# apply reaction force
vector.copy_p(fint, fext)

# residual
fres = fext - fint
# print residual
assert np.isclose(np.sum(np.abs(fres)) / np.sum(np.abs(fext)), 0,  atol=1e-6)
# plot
# ----
parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Plot result")
parser.add_argument("--save", type=str, help="Save plot (plot not shown)")
args = parser.parse_args(sys.argv[1:])

if args.plot:
    import GooseMPL as gplt
    import matplotlib.pyplot as plt

    plt.style.use(["goose", "goose-latex"])

    # average equivalent stress per element
    dV = elem.AsTensor(2, elem.dV)
    Sigav = np.average(mat.Sig, weights=dV, axis=1)
    sigeq = GMatElastic.Cartesian3d.Sigeq(Sigav)

    # plot
    fig, ax = plt.subplots()
    gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq, cmap="jet", axis=ax, clim=(0, 0.1))
    gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)

    # optional save
    if args.save is not None:
        fig.savefig(args.save)
    else:
        plt.show()

    plt.close(fig)
