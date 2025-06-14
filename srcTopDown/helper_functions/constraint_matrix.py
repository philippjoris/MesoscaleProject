"""

UNDER CONSTRUCTION - NOT USABLE YET

A function for periodic boundary conditions.

"""

import numpy as np
import random

mesh = GooseFEM.Mesh.Quad4.Regular(5, 5)

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


def contraint_matrix(mesh):
    mesh.nodesPeriodic
    mesh.nodesOrigin
    mesh.nodesBottomLeftCorner
    mesh.nodesBottomRightCorner
    mesh.nodesTopLeftCorner
    mesh.nodesTopRightCorner

    return constraintsMatrix