import argparse
import sys

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import GooseFEM
import numpy as np
import random
# mesh
# ----

# define mesh
print("running a GooseFEM static PBC example...")
mesh = GooseFEM.Mesh.Quad4.Regular(20, 20)
meshRefined = GooseFEM.Mesh.Quad4.Map.RefineRegular(mesh, 5, 5)

# mesh dimensions
nelemCoarse = meshRefined.coarseMesh.nelem
nelem = meshRefined.fineMesh.nelem
nne = meshRefined.fineMesh.nne
ndim = meshRefined.fineMesh.ndim
tyinglist = meshRefined.fineMesh.nodesPeriodic

# mesh definition
coor = meshRefined.fineMesh.coor
conn = meshRefined.fineMesh.conn
dofs = meshRefined.fineMesh.dofs

# create control nodes
control = GooseFEM.Tyings.Control(coor, dofs)

# add control nodes
coor = control.coor

# list of prescribed DOFs (fixed node + control nodes)
iip = np.concatenate((
    dofs[np.array([mesh.nodesBottomLeftCorner]), 0],
    dofs[np.array([mesh.nodesBottomLeftCorner]), 1],
    control.controlDofs[0],
    control.controlDofs[1]
))

# initialize my periodic boundary condition class
periodicity = GooseFEM.Tyings.Periodic(coor, control.dofs, control.controlDofs, tyinglist, iip)
dofs = periodicity.dofs

# simulation variables
# --------------------

# vector definition
vector = GooseFEM.VectorPartitionedTyings(conn, dofs, periodicity.Cdu, periodicity.Cdp, periodicity.Cdi)

# element definition
elem0 = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor))
elem = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor))
nip = elem.nip

# nodal quantities
disp = np.zeros_like(coor)
du = np.zeros_like(coor)  # iterative displacement update
fint = np.zeros_like(coor)  # internal force
fext = np.zeros_like(coor)  # external force

# element vectors / matrix
ue = vector.AsElement(disp)
coore = vector.AsElement(coor)
fe = np.empty([nelem, nne, ndim])
Ke = np.empty([nelem, nne * ndim, nne * ndim])

# DOF values
Fext = np.zeros([periodicity.nni])
Fint = np.zeros([periodicity.nni])

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
tauy0 = randomizeMicrostr(nelemCoarse, nip, 0.7, .600, .200)
tauy0Fine = meshRefined.mapToFine(tauy0)
mat = GMat.LinearHardening2d(K=np.ones([nelem, nip])*170, G=np.ones([nelem, nip])*80, tauy0=tauy0Fine, H=np.ones([nelem, nip])*0.2)

# allocate system matrix
K = GooseFEM.MatrixPartitionedTyings(conn, dofs, periodicity.Cdu, periodicity.Cdp)
Solver = GooseFEM.MatrixPartitionedTyingsSolver()

# array of unit tensor
I2 = GMatTensor.Cartesian3d.Array2d(mat.shape).I2

# solve
# -----
ninc = 501
max_iter = 50
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

# ue = vector.AsElement(disp)
# du = np.zeros_like(disp)
initial_guess = np.zeros_like(disp)
total_increment = np.zeros_like(disp)

# deformation gradient
F = np.array(
        [
            [1.0 + (0.2/ninc), 0.0],
            [0.0, 1.0 / (1.0 + (0.2/ninc))]
        ]
    )

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):

    converged = False

    mat.increment()

    disp += initial_guess
    total_increment = initial_guess.copy()
    for iter in range(max_iter):  
        # deformation gradient
        vector.asElement(disp, ue)
        elem0.symGradN_vector(ue, mat.F)
        mat.F += I2
        mat.refresh(tangent)  

        # internal force
        elem.int_gradN_dot_tensor2_dV(mat.Sig, fe)
        vector.assembleNode(fe, fint)

        # stiffness matrix
        elem.int_gradN_dot_tensor4_dot_gradNT_dV(mat.C, Ke)
        K.assemble(Ke)

        # residual
        fres = fext - fint

        if iter > 0:
            # - internal/external force as DOFs (account for periodicity)
            vector.asDofs_i(fext, Fext)
            vector.asDofs_i(fint, Fint)
            # - extract reaction force
            vector.copy_p(Fint, Fext)
            # - norm of the residual and the reaction force
            nfres = np.sum(np.abs(Fext - Fint))
            nfext = np.sum(np.abs(Fext))
            # - relative residual, for convergence check
            if nfext:
                res = nfres / nfext
            else:
                res = nfres
            # print (f"Iter {iter}, Residual = {res_norm}")
            if iter > 10:
                a = 1
            if res < 1e-06:
                print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res}")
                converged = True
                break

        du.fill(0.0)

        # initialise displacement update
        if iter == 0:
            du[control.controlNodes, 0] = (F[0,:] - np.eye(2)[0, :]) 
            du[control.controlNodes, 1] = (F[1,:] - np.eye(2)[1, :])  

        # solve
        Solver.solve(K, fres, du)
        
        # add delta u
        disp += du
        total_increment += du

        elem.update_x(vector.AsElement(coor + disp))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))
    
    if converged:
         # print(total_increment)
         initial_guess = 0.2 * total_increment
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")


# post-process
# ------------
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

    # plt.style.use(["goose", "goose-latex"])

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
    ax.set_xlim(-10,125)
    # ax.set_ylim(0,110)
    
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
    ax.set_xlim(-10,125)
    # ax.set_ylim(0,110)
    
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