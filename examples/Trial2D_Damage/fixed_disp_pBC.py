import argparse
import sys
import os

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import GooseFEM
import numpy as np
import random

from srcTopDown.helper_functions.element_erosion import element_erosion

script_dir = os.path.dirname(os.path.abspath(__file__))

# mesh
# ----
refinement_nr = 2

# define mesh
print("running a GooseFEM static PBC example...")
mesh = GooseFEM.Mesh.Quad4.Regular(40, 40)
meshRefined = GooseFEM.Mesh.Quad4.Map.RefineRegular(mesh,refinement_nr, refinement_nr)

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
vector = GooseFEM.VectorPartitionedTyings(dofs, periodicity.Cdu, periodicity.Cdp, periodicity.Cdi)

# element definition
elem0 = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, conn))
elem = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, conn))
nip = elem.nip

# nodal quantities
disp = np.zeros_like(coor)
du = np.zeros_like(coor)  # iterative displacement update
fint = np.zeros_like(coor)  # internal force
fext = np.zeros_like(coor)  # external force

# element vectors / matrix
ue = vector.AsElement(disp, conn)
coore = vector.AsElement(coor, conn)
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
mat = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem, nip])*170,
    G=np.ones([nelem, nip])*80,
    tauy0=tauy0Fine,
    H=np.ones([nelem, nip])*0.2,
    D1=np.ones([nelem, nip])*0.1,
    D2=np.ones([nelem, nip])*0.2,
    D3=np.ones([nelem, nip])*-1.7
    )
# allocate system matrix
K = GooseFEM.MatrixPartitionedTyings(dofs, periodicity.Cdu, periodicity.Cdp)
Solver = GooseFEM.MatrixPartitionedTyingsSolver()

# array of unit tensor
I2 = GMatTensor.Cartesian3d.Array2d(mat.shape).I2

# solve
# -----
ninc = 801
max_iter = 80
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
            [1.0 + (0.4/ninc), 0.0],
            [0.0, 1.0 / (1.0 + (0.4/ninc))]
        ]
    )

stop_simulation = False

failed_elem = set()

for ilam in range(ninc):

    converged = False

    mat.increment()

    disp += initial_guess
    total_increment = initial_guess.copy()
    for iter in range(max_iter):  
        # deformation gradient
        ue = vector.AsElement(disp, conn)
        elem0.symGradN_vector(ue, mat.F)
        mat.F += I2
        mat.refresh(tangent, element_erosion=True) 

        # internal force
        elem.int_gradN_dot_tensor2_dV(mat.Sig, fe)
        fint = vector.AssembleNode(fe, conn)

        # stiffness matrix
        elem.int_gradN_dot_tensor4_dot_gradNT_dV(mat.C, Ke)
        K.clear()
        K.assemble(Ke, conn)
        K.finalize()

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
            if res < 1e-06:
                print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res}")
                converged = True
                
                if (np.amax(mat.D_damage) < 1):
                    break
                else:
                    curr_failed = failed_elem.copy()
                    # delete cells with damage > 1 and append to vector
                    cellDamage = meshRefined.averageToCoarse(mat.D_damage, np.ones_like(mat.D_damage))
                    for k, obj in enumerate(cellDamage):
                        if any(IP >= 1 for IP in obj):
                            curr_failed.add(k)
                    #for k, obj in enumerate(mat.D_damage):
                    #    if any(IP >= 1 for IP in obj):
                    #        curr_failed.add(k)

                    newly_failed = curr_failed - failed_elem
                    failed_elem = curr_failed.copy()

                    if len(newly_failed) > 7:
                        for failed in newly_failed:
                            print(f"this are the newly failed elements {newly_failed}")
                            for elem in meshRefined.map[failed]:
                                mat.delete_element(failed)
                        mat.refresh(tangent, element_erosion=True) 
                        break
                    elif newly_failed:
                        print(f"INFO: Elements {newly_failed} failed.")
                        # erode element and set strain and stress to zero
                        # -----------------------
                        #
                        #   multiple elements deleted at onces, what happens to the forces ?
                        #   The following algorithm is only for single element deletions.
                        #
                        # -----------------------
                        disp, elem, mat = element_erosion(Solver, vector, coor, mat, elem, elem0, fe, fext, disp, newly_failed, K, fe, I2, meshRefined)                    
                    break

        du.fill(0.0)

        # initialise displacement update
        if iter == 0:
            du[control.controlNodes, 0] = (F[0,:] - np.eye(2)[0, :]) 
            du[control.controlNodes, 1] = (F[1,:] - np.eye(2)[1, :])  

        # solve
        Solver.solve(K, fres, du)
        
        disp += du
        total_increment += du        
        
        elem.update_x(vector.AsElement(coor + disp, conn))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))
    

    # ----------------------------
    # break if more than 15 elements were deleted
    if len(failed_elem) > 25:
        break
    # ----------------------------

    if converged:
        initial_guess = 0.3 * total_increment                 
    else:
        print (f"WARNING: Increment {ilam}/{ninc} DID NOT converged at Iter {iter}, Residual = {res}")
        stop_simulation = True
        raise RuntimeError(f"Load step {ilam} failed to converge.")


# post-process
# ------------
# plot
# ----
parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Plot result")
parser.add_argument("--save", action="store_true", help="Save plot (plot not shown)")
args = parser.parse_args(sys.argv[1:])


coarseCoorDisp = np.zeros_like(meshRefined.coarseMesh.coor)
for i in range(len(coarseCoorDisp)):
    coarseCoorDisp[i][:] = coor[refinement_nr*i][:] + disp[refinement_nr*i][:]

if args.plot:
    import GooseMPL as gplt
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plt.style.use(["goose", "goose-latex"])

    # Average quantities fine mesh
    dV = elem.AsTensor(2, elem.dV)
    Sigav = np.average(mat.Sig, weights=dV, axis=1)
    sigeq_av = GMat.Sigeq(Sigav)
    epseq_av = GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1))
    damage_av = np.average(mat.D_damage, axis = 1)
    # Average quantities coarse mesh
    sigCoarsePlot = meshRefined.averageToCoarse(sigeq_av, np.ones([nelem]))
    epsCoarsePlot = meshRefined.averageToCoarse(epseq_av, np.ones([nelem]))
    damageCoarsePlot = meshRefined.averageToCoarse(damage_av, np.ones([nelem]))

    # plot stress
    fig, ax = plt.subplots(figsize=(8, 6))
    # gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq_av, cmap="jet", axis=ax)
    gplt.patch(coor=coor+disp, conn=meshRefined.fineMesh.conn, cindex=sigeq_av, cmap="jet", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(sigeq_av)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent mises stress")

    # optional save
    if args.save:
        fig.savefig(os.path.join(script_dir, 'fixed-disp_contour_sig.pdf'))
    else:
        plt.show()

    plt.close(fig)

    # plot strain
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn, cindex=epseq_av, cmap="jet", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(epseq_av)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent strain")

    # optional save
    if args.save:
        fig.savefig(os.path.join(script_dir, 'fixed-disp_contour_eps.pdf'))
    else:
        plt.show()

    plt.close(fig)

    # plot Damage
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn, cindex=damage_av, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(damage_av)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Damage")

    # optional save
    if args.save:
        fig.savefig(os.path.join(script_dir, 'fixed-disp_damage.pdf'))
    else:
        plt.show()

    plt.close(fig)    

    # plot
    fig, ax = plt.subplots()
    ax.plot(epseq, sigeq, c="r", label=r"LinearHardening")

    # optional save
    if args.save:
        fig.savefig(os.path.join(script_dir, 'fixed-disp_sig-eps.pdf'))
    else:
        plt.show()

    plt.close(fig)