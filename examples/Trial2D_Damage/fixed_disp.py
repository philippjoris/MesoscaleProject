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
elem0 = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor))
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
mat = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem, nip])*170,
    G=np.ones([nelem, nip])*80,
    tauy0=tauy0,
    H=np.ones([nelem, nip])*0.2,
    D1=np.ones([nelem, nip])*0.1,
    D2=np.ones([nelem, nip])*0.2,
    D3=np.ones([nelem, nip])*-1.7
    )
I2 = GMatTensor.Cartesian3d.Array2d(mat.shape).I2  
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

du = np.zeros_like(disp)
du_noInitial = np.zeros_like(disp)
du_last = np.zeros_like(vector.AsDofs_u(disp))

initial_guess = np.zeros_like(disp)
failed_elem = np.array([])
# xp = np.zeros_like(vector.AsDofs_p(disp))
for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):

    # update displacement
    du[mesh.nodesRightEdge, 0] = (+1.0/ninc)
    du[mesh.nodesTopEdge, 1] = (-0.4/ninc)
    du[mesh.nodesLeftEdge, 0] = 0.0  # not strictly needed: default == 0
    du[mesh.nodesBottomEdge, 1] = 0.0  # not strictly needed: default == 0
    

    # convergence flag
    converged = False
    
    mat.increment()

    # disp += vector.NodeFromPartitioned(du_last, vector.AsDofs_p(du))
    # elem.update_x(vector.AsElement(coor + disp))
    total_increment = du_last.copy()
    # du_noInitial.fill(0.0)
    for iter in range(max_iter): 
        # update element wise displacments
        ue = vector.AsElement(disp) 

        # update deformation gradient F
        elem0.symGradN_vector((ue), mat.F)
        mat.F += I2
        mat.refresh(tangent)  
  
        # update internal forces and assemble
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        fint = vector.AssembleNode(fe)

        # update stiffness matrix
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
        K.assemble(Ke)

        fres = - fint

        if iter > 0:
            # residual 
            fres_u = -vector.AsDofs_u(fint)
            
            res_norm = np.linalg.norm(fres_u) 
            # print (f"Iter {iter}, Residual = {res_norm}")
            if res_norm < 1e-06:
                print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
                converged = True
                if (np.amax(mat.D_damage) < 1):
                    break
                else:
                    # delete cells with damage > 1 and append to vector
                    fext_curr = fext.copy()
                    fe_curr = fe.copy()
                    disp_curr = disp.copy()

                    for k, obj in enumerate(mat.D_damage):
                        if any(IP>=1 for IP in obj):
                            failed_elem = np.append(failed_elem, k)

                    failed_elem = np.array(failed_elem, dtype=int)

                    # create new external force vector per element
                    new_fext_elem = np.zeros_like(fe)

                    for failed in failed_elem:
                        mat.delete_element(failed)                    
                    
                    for incr in np.linspace(1.0, 0.0, 20):
                        for failed in failed_elem:                     
                            new_fext_elem[failed] = -incr * fe_curr[failed]

                        # create new external force vector per node                            
                        fext = fext_curr + vector.AssembleNode(new_fext_elem)

                        forces_smoothed = False
                        disp = disp_curr.copy()
                        
                        for second_iter in range(max_iter):
                            vector.asElement(disp, ue)
                            elem0.symGradN_vector(ue, mat.F)
                            mat.F += I2
                            mat.refresh(tangent, element_erosion=True) 

                            # internal force
                            elem.int_gradN_dot_tensor2_dV(mat.Sig, fe)
                            
                            vector.assembleNode(fe, fint)

                            # stiffness matrix
                            elem.int_gradN_dot_tensor4_dot_gradNT_dV(mat.C, Ke)
                            K.assemble(Ke)

                            fres_load = fext - fint
                            Fext = vector.AsDofs_u(fext)
                            Fint = vector.AsDofs_u(fint)   
                            nfres = np.sum(np.abs(Fext - Fint))                 
                            nfext = np.sum(np.abs(Fext))
                            if nfext:
                                res = nfres / nfext
                            else:
                                res = nfres                    
                            if res < 1e-06:
                                forces_smoothed = True
                                break

                            du.fill(0.0)
                            Solver.solve(K, fres_load, du)
                            disp += du
                            elem.update_x(vector.AsElement(coor + disp))

                        if not forces_smoothed:
                            raise RuntimeError(f"Free forces due to element erosion not smoothed successfully.")                            
                    # erode element and set strain and stress to zero
                    break

            # solve
            du.fill(0.0)

        Solver.solve(K, fres, du)
        # add newly found delta_u to total increment
        total_increment += vector.AsDofs_u(du)
        # update displacement vector
        disp += du

        # update shape functions
        elem.update_x(vector.AsElement(coor + disp))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))
    
    if converged:
        continue
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")


# post-process
# ------------
# strain
elem0.symGradN_vector((coore + ue), mat.F)
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