import argparse
import sys

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import GooseFEM
import numpy as np
import random
#
# Example with 2D varying microstructure and fixed disp
#
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
        dofs[mesh.nodesRightEdge, 0],
        dofs[mesh.nodesTopEdge, 1],
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
def randomizeMicrostr(nelem, nip, fraction_soft, value_hard, value_soft):
    array = np.ones([nelem, nip])*value_hard
    nsoft = round(fraction_soft * nelem)
    softelem = random.sample(range(nelem), k=nsoft)
    for elem in softelem:
        array[elem] *= (value_soft/value_hard)
    return array
# -------------------
# mat = GMat.Elastic2d(K=np.ones([nelem, nip]), G=np.ones([nelem, nip]))
tauy0 = randomizeMicrostr(nelem, nip, 0.7, .600, .200)
mat = GMat.LinearHardening2d(K=np.ones([nelem, nip])*170, G=np.ones([nelem, nip])*80, tauy0=tauy0, H=np.ones([nelem, nip])*1)

# simulation variables
# --------------------
ue = vector.AsElement(disp)
coore = vector.AsElement(coor)
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

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

curr_increment = np.zeros_like(vector.AsDofs_u(disp))
total_increment = np.zeros_like(vector.AsDofs_u(disp))

initial_guess = np.zeros_like(vector.AsDofs_u(disp))
# xp = np.zeros_like(vector.AsDofs_p(disp))
for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    # store old displacement
    xp = vector.AsDofs_p(disp).copy()

    # update displacement
    disp[mesh.nodesRightEdge, 0] += (+2.0/ninc)
    disp[mesh.nodesTopEdge, 1] += (-0.8/ninc)
    disp[mesh.nodesLeftEdge, 0] = 0.0  # not strictly needed: default == 0
    disp[mesh.nodesBottomEdge, 1] = 0.0  # not strictly needed: default == 0
    
    # new displacement increment
    delta_xp = vector.AsDofs_p(disp) - xp

    # convergence flag
    converged = False

    mat.increment()

    # impose initial guess on unknown displacements
    disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + initial_guess, vector.AsDofs_p(disp))

    total_increment[:] = 0.
    for iter in range(max_iter): 
        # update element wise displacments
        ue = vector.AsElement(disp) 

        # update deformation gradient F
        elem.gradN_vector((coore + ue), mat.F)
        mat.F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
        mat.refresh(tangent)  
  
        # update internal forces and assemble
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        fint = vector.AssembleNode(fe)

        # residual 
        fres = -vector.AsDofs_u(fint)
        
        res_norm = np.linalg.norm(fres) 
        # print (f"Iter {iter}, Residual = {res_norm}")
        if res_norm < 1e-06:
            print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
            converged = True
            break

        # update stiffness matrix
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
        K.assemble(Ke)

        # solve
        #if iter == 0:
        #    Solver.solve_u(K, fres, delta_xp, curr_increment)
        #else:
        #    Solver.solve_u(K, fres, np.zeros_like(xp), curr_increment)
        Solver.solve_u(K, fres, np.zeros_like(xp), curr_increment)
        # add newly found delta_u to total increment
        total_increment += curr_increment
        # update displacement vector
        disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + curr_increment, vector.AsDofs_p(disp))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))
    
    if converged:
        initial_guess = initial_guess + total_increment
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")


print(fint)
# post-process
# ------------
# strain
elem.gradN_vector((coore + ue), mat.F)
mat.F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
mat.refresh()  

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
parser.add_argument("--save", action="store_true", help="Save plot (plot not shown)")
args = parser.parse_args(sys.argv[1:])

if args.plot:
    import GooseMPL as gplt
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plt.style.use(["goose", "goose-latex"])

    # Average equivalent stress per element
    dV = elem.AsTensor(2, elem.dV)
    Sigav = np.average(mat.Sig, weights=dV, axis=1)
    sigeq_av = GMat.Sigeq(Sigav)
    # Average eq. strain per element
    epseq_av = GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1))

    # plot stress
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq_av, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    ax.set_xlim(0,130)
    ax.set_ylim(0,110)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(sigeq_av)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent mises stress")

    # optional save
    if args.save:
        fig.savefig('fixed-disp_contour_sig.pdf')
    else:
        plt.show()

    plt.close(fig)

    # plot strain
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn, cindex=epseq_av, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    ax.set_xlim(0,130)
    ax.set_ylim(0,110)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(epseq_av)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent strain")

    # optional save
    if args.save:
        fig.savefig('fixed-disp_contour_eps.pdf')
    else:
        plt.show()

    plt.close(fig)

    # plot
    fig, ax = plt.subplots()
    ax.plot(epseq, sigeq, c="r", label=r"LinearHardening")

    # optional save
    if args.save:
        fig.savefig('fixed-disp_sig-eps.pdf')
    else:
        plt.show()

    plt.close(fig)