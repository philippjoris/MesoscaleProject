import argparse
import sys

import GMatElastic
import GooseFEM
import numpy as np


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
        dofs[mesh.nodesRight, 0],
        dofs[mesh.nodesTop, 1],
        dofs[mesh.nodesLeft, 0],
        dofs[mesh.nodesBottom, 1],
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

mat = GMatElastic.Cartesian3d.Elastic2d(K=np.ones([nelem, nip]), G=np.ones([nelem, nip]))

# solve
# -----

# strain
ue = vector.AsElement(disp)
elem.symGradN_vector(ue, mat.Eps)
mat.refresh()

# internal force
fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
fint = vector.AssembleNode(fe)

# stiffness matrix
Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
K.assemble(Ke)

# residual
fres = fext - fint

# set fixed displacement
disp[mesh.nodesRight, 0] = +1.0
disp[mesh.nodesTop, 1] = -1.0
disp[mesh.nodesLeft, 0] = 0.0  # not strictly needed: default == 0
disp[mesh.nodesBottom, 1] = 0.0  # not strictly needed: default == 0

# solve
Solver.solve(K, fres, disp)

# post-process
# ------------

# strain
vector.asElement(disp, ue)
elem.symGradN_vector(ue, mat.Eps)
mat.refresh()

# internal force
elem.int_gradN_dot_tensor2_dV(mat.Sig, fe)
vector.assembleNode(fe, fint)

# apply reaction force
vector.copy_p(fint, fext)

# residual
fres = fext - fint

# print residual
assert np.isclose(np.sum(np.abs(fres)) / np.sum(np.abs(fext)), 0)

# plot
# ----
parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Plot result")
parser.add_argument("--save", type=str, help="Save plot (plot not shown)")
args = parser.parse_args(sys.argv[1:])

if args.plot:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    from matplotlib.cm import ScalarMappable
    import numpy as np

    plt.style.use(["goose", "goose-latex"])

    # Average equivalent stress per element
    dV = elem.AsTensor(2, elem.dV)
    Sigav = np.average(mat.Sig, weights=dV, axis=1)
    sigeq = GMatElastic.Cartesian3d.Sigeq(Sigav)

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    norm = plt.Normalize(vmin=0)
    colors = cmap(norm(sigeq))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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
    mappable.set_array(sigeq)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent stress")

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])  # Ensure scaling works properly

    # Optional save or show
    if args.save is not None:
        fig.savefig(args.save)
    else:
        plt.show()

    plt.close(fig)