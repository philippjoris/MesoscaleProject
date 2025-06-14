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
mesh = GooseFEM.Mesh.Quad4.Regular(10, 10)

# mesh dimensions
nelem = mesh.nelem
nne = mesh.nne
ndim = mesh.ndim
tyinglist = mesh.nodesPeriodic

# mesh definition
coor = mesh.coor
conn = mesh.conn
dofs = mesh.dofs

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

# initialize displacements
# initialize forces
disp = np.zeros_like(coor, dtype=float)
fext = np.zeros_like(coor)

# simulation variables
# --------------------

# vector definition
vector = GooseFEM.VectorPartitionedTyings(conn, dofs, periodicity.Cdu, periodicity.Cdp, periodicity.Cdi)

# allocate system matrix
K = GooseFEM.MatrixPartitionedTyings(conn, dofs, periodicity.Cdu, periodicity.Cdp)
Solver = GooseFEM.MatrixPartitionedTyingsSolver()


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

# here I have to pass the displacements without the extra control DOFs
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
ninc = 701
max_iter = 50
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

curr_increment = np.zeros_like(periodicity.iiu)
total_increment = np.zeros_like(periodicity.iiu)

initial_guess = np.zeros_like(periodicity.iiu)
# xp = np.zeros_like(vector.AsDofs_p(disp))

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):

    # update deformation gradient
    F = np.array(
        [
            [1.0 + (0.2*lam), 0.0],
            [0.0, 1.0 / (1.0 + (0.2*lam))]
        ]
    )
    disp.ravel()[periodicity.iip[0]] = 0.0
    disp.ravel()[periodicity.iip[1]] = 0.0
    disp.ravel()[periodicity.control[0]] = (F[0,:] - np.eye(2)[0, :]) 
    disp.ravel()[periodicity.control[1]] = (F[1,:] - np.eye(2)[1, :]) 

    disp.ravel()[periodicity.iiu] += initial_guess
    disp.ravel()[periodicity.iip] = disp.ravel()[periodicity.iip]
    disp.ravel()[periodicity.iid] = periodicity.Cdi @ disp.ravel()[periodicity.iii]
    # convergence flag
    converged = False

    mat.increment()

    total_increment[:] = 0.
    for iter in range(max_iter): 
        # update element wise displacments
        ue = vector.AsElement(disp.ravel())

        # update deformation gradient F
        elem.gradN_vector((coore + ue), mat.F)
        mat.F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
        mat.refresh(tangent)  
  
        # update internal forces and assemble
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        fint = vector.AssembleNode(fe)

        # RHS
        r = -fint.ravel()
        ru = r.ravel()[periodicity.iiu]
        rd = r.ravel()[periodicity.iid]
        res_norm = np.linalg.norm(ru) 
        # print (f"Iter {iter}, Residual = {res_norm}")
        if res_norm < 1e-06:
            print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
            converged = True
            break

        # update stiffness matrix
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
        K.assemble(Ke)

        print(ru)
        Solver.solve_u(K, ru, disp.ravel()[periodicity.iid], disp.ravel()[periodicity.iip], curr_increment)

        # reconstruct node vector
        disp.ravel()[periodicity.iiu] += curr_increment
        disp.ravel()[periodicity.iid] = periodicity.Cdi @ disp.ravel()[periodicity.iii]

        # add newly found delta_u to total increment
        total_increment += curr_increment

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))
    
    if converged:
        initial_guess = initial_guess + total_increment
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")


# print(disp)
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