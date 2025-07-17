import argparse
import sys

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import GooseFEM
import numpy as np
from scipy.sparse.linalg import cg
# mesh
# ----

# define mesh
print("running a GooseFEM static example...")
mesh = GooseFEM.Mesh.Hex8.Regular(5, 5, 5)

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
        # dofs[mesh.nodesRight, 0],
        # dofs[mesh.nodesTop, 1],
        dofs[mesh.nodesLeft, 0],
        dofs[mesh.nodesLeft, 1],
        dofs[mesh.nodesLeft, 2],                
        #dofs[mesh.nodesBottom, 1],
        #dofs[mesh.nodesFront, 2],
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
elem = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor))
nip = elem.nip

# material definition
# -------------------

# mat = GMat.Elastic2d(K=np.ones([nelem, nip]), G=np.ones([nelem, nip]))
mat = GMat.LinearHardening2d(K=np.ones([nelem, nip])*170, G=np.ones([nelem, nip])*80, tauy0=np.ones([nelem, nip])*.600, H=np.ones([nelem, nip])*1)

# solve
# -----

# strain
ue = vector.AsElement(disp)
coore = vector.AsElement(coor)
elem.gradN_vector((coore+ue), mat.F)
mat.refresh()

# internal force of the right hand side per element and assembly
fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
fint = vector.AssembleNode(fe)

# stiffness matrix
Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
K.assemble(Ke)

# residual
fres = fext - fint

# solve
# -----
ninc = 501
max_iter = 70
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

curr_increment = np.zeros_like(vector.AsDofs_u(disp))
total_increment = np.zeros_like(vector.AsDofs_u(disp))

initial_guess = np.zeros_like(vector.AsDofs_u(disp))
for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    # store old displacement
    xp = vector.AsDofs_p(disp).copy()
    # introduce new displacement increment
    fext[131, 0] += (+8.0/ninc)
    # disp[mesh.nodesTop, 1] += (-0.5/ninc)
    disp[mesh.nodesLeft, 0] = 0.0  # not strictly needed: default == 0
    disp[mesh.nodesLeft, 1] = 0.0  # not strictly needed: default == 0
    disp[mesh.nodesLeft, 2] = 0.0  # not strictly needed: default == 0        
    # disp[mesh.nodesBottom, 1] = 0.0  # not strictly needed: default == 0
    # dofs[mesh.nodesFront, 2] = 0.0

    # new displacement increment
    delta_xp = vector.AsDofs_p(disp) - xp

    # convergence flag
    converged = False

    mat.increment()

    # impose initial guess on unknown displacements
    disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + initial_guess, vector.AsDofs_p(disp))

    # curr_increment = np.zeros_like(vector.AsDofs_u(disp))
    total_increment = initial_guess.copy()
    for iter in range(max_iter): 
        # update element wise displacments
        ue = vector.AsElement(disp) 

        # update deformation gradient F
        elem.gradN_vector((coore + ue), mat.F)
        mat.refresh(tangent)  

        # update internal forces and assemble
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        fint = vector.AssembleNode(fe)

        # residual 
        fres = vector.AsDofs_u(fext) - vector.AsDofs_u(fint)

        res_norm = np.linalg.norm(fres) 

        if res_norm < 1e-06:
            print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
            converged = True
            break
        if iter > 5:
            a=1
        # update stiffness matrix
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
        K.assemble(Ke)

        # solve
        Solver.solve_u(K, fres, np.zeros_like(xp), curr_increment)

        # add newly found delta_u to total increment
        total_increment += curr_increment
        
        # update displacement vector
        disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + curr_increment, vector.AsDofs_p(disp))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))

    if converged:
        initial_guess = total_increment
    if not converged:
        print(f"WARNING: not converge at displacement {0.4*ilam/ninc} but no Runtime termination.")
        break
        # raise RuntimeError(f"Load step {ilam} failed to converge.")

# post-process
# ------------
# strain
elem.gradN_vector((coore + ue), mat.F)
# mat.F[:,:,0,0] += 1.0
# mat.F[:,:,1,1] += 1.0
# mat.F[:,:,2,2] += 1.0
mat.refresh()  

# internal force
elem.int_gradN_dot_tensor2_dV(mat.Sig, fe)
vector.assembleNode(fe, fint)

# apply reaction force
vector.copy_p(fint, fext)

# residual
fres = vector.AsDofs_u(fext) - vector.AsDofs_u(fint)
# print residual
#assert np.isclose(np.sum(np.abs(fres)), 0,  atol=3e-6)

# plot
# ----
parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Plot result")
parser.add_argument("--save", action="store_true", help="Save plot (plot not shown)")
args = parser.parse_args(sys.argv[1:])

if args.plot:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FormatStrFormatter

    plt.style.use(["goose", "goose-latex"])

    # Average equivalent stress per element
    dV = elem.AsTensor(2, elem.dV)
    Sigav = np.average(mat.Sig, weights=dV, axis=1)
    sigeq_av = GMat.Sigeq(Sigav)
    epseq_av = GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    # if all values are equal, extend range a bit:
    vmax = sigeq_av.max()
    vmin = 0

    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(sigeq_av))

    faces = [
    [0, 1, 2, 3],  # bottom face
    [4, 5, 6, 7],  # top face
    [0, 1, 5, 4],  # front face
    [1, 2, 6, 5],  # right face
    [2, 3, 7, 6],  # back face
    [3, 0, 4, 7]   # left face
    ]

    # Plot deformed mesh with colors
    for i, element in enumerate(conn):
        verts = np.array(coor[element] + disp[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts], facecolors=[colors[i]], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(poly)

    # Plot undeformed mesh with dashed edges and transparent faces
    for element in conn:
        verts = np.array(coor[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts],
                                    facecolors=[[0, 0, 0, 0]],  # Fully transparent
                                    edgecolors='k',
                                    linewidths=0.5,
                                    linestyles='dashed')
            ax.add_collection3d(poly)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(sigeq_av)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent stress")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])  # Ensure scaling works properly

    # Optional save or show
    if args.save:
        fig.savefig('fixed-disp_contour_sig.pdf')
    else:
        plt.show()

    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    # if all values are equal, extend range a bit:
    vmax = epseq_av.max()
    vmin = 0

    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(epseq_av))
    
    faces = [
    [0, 1, 2, 3],  # bottom face
    [4, 5, 6, 7],  # top face
    [0, 1, 5, 4],  # front face
    [1, 2, 6, 5],  # right face
    [2, 3, 7, 6],  # back face
    [3, 0, 4, 7]   # left face
    ]

    # Plot deformed mesh with colors
    for i, element in enumerate(conn):
        verts = np.array(coor[element] + disp[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts], facecolors=[colors[i]], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(poly)

    # Plot undeformed mesh with dashed edges and transparent faces
    for element in conn:
        verts = np.array(coor[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts],
                                    facecolors=[[0, 0, 0, 0]],  # Fully transparent
                                    edgecolors='k',
                                    linewidths=0.5,
                                    linestyles='dashed')
            ax.add_collection3d(poly)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(epseq_av)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent strain")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])  # Ensure scaling works properly

    # Optional save or show
    if args.save:
        fig.savefig('fixed-disp_contour_eps.pdf')
    else:
        plt.show()

    plt.close(fig)        
