import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import numpy as np
import meshio
import os

# A dictionary to map keywords to plotting parameters
PLOT_CONFIG = {
    "stress": {
        "data_key": "stress", 
        "label": "Equivalent stress", 
        "format": '%.1f',
        "filename": 'results_contour_sig.pdf',
        "location": "element"
    },
    "strain": {
        "data_key": "strain", 
        "label": "Equivalent strain", 
        "format": '%.2f',
        "filename": 'results_contour_eps.pdf',
        "location": "element"        
    },
    "damage": {
        "data_key": "damage", 
        "label": "Damage", 
        "format": '%.2f',
        "filename": 'results_contour_damage.pdf',
        "location": "element"        
    },
    "triaxiality": {
        "data_key": "triaxiality", 
        "label": "Stress triaxiality", 
        "format": '%.2f',
        "filename": 'results_contour_triaxiality.pdf',
        "location": "element"        
    },
    "failed_elements": {
        "data_key": "failed", 
        "label": "Failed", 
        "format": '%.2f',
        "filename": 'results_failed_elems.pdf',
        "location": "element"        
    }    
    # Future keywords can be added here
}


def _ensure_list(obj):
    """If obj is not a list/tuple/np.ndarray, wraps it in a list."""
    # Check if the object is iterable and not a string/bytes
    if isinstance(obj, (list, tuple, np.ndarray)):
        # If it's iterable, we need to check if it's a list of objects 
        # (the desired input) or just a single array (e.g., coor or disp).
        # Since elem, mat, conn are expected to contain objects, 
        # we check for basic iterable types.
        return obj
    else:
        # If it's a single object (not iterable), wrap it in a list
        return [obj]
    
def prepare_plot_data(elem, mat, conn, coor, disp):
    """
    Calculates element-wise average values and prepares data for 3D plotting.

    Args:
        elem (list): List of element objects (must have dV property).
        mat (list): List of material objects (must have Sig, F, D_damage, Sig_triax properties).
        conn (list): List of connectivity arrays.
        coor (np.array): Nodal coordinates.
        disp (np.array): Nodal displacements.

    Returns:
        dict: A dictionary containing the averaged data arrays and mesh info.
    """
    
    # Imports needed for GMat.Sigeq, GMat.Epseq, GMat.Strain - assuming GMat is a module
    # with these methods. If it's a class, adjust accordingly.
    # from your_module import GMat 

    elem_list = _ensure_list(elem)
    mat_list = _ensure_list(mat)
    # conn_list = conn if conn.ndim > 2 else np.expand_dims(conn,axis=0)

    sigeq_av_all = []
    epseq_av_all = []
    damage_av_all = []
    sig_triax_av_all = []
    failed_all = []
    conn_all = []

    for e, m, c in zip(elem_list, mat_list, conn):
        if e is None or m is None or len(c) == 0:
            continue
        dV = e.AsTensor(2, e.dV)
        
        # Stress components are often available directly on the material object
        Sigav = np.average(m.Sig, weights=dV, axis=1) 
        
        sigeq_av_all.append(GMat.Sigeq(Sigav))
        epseq_av_all.append(
            GMat.Epseq(np.average(GMat.Strain(m.F), axis=1))
        )
        damage_av_all.append(np.average(m.D_damage, axis=1))
        sig_triax_av_all.append(np.average(m.Sig_triax, axis=1))
        failed_all.append(np.any(m.failed, axis=1).astype(float))

        conn_all.append(c)

    return {
        "stress": np.concatenate(sigeq_av_all, axis=0),
        "strain": np.concatenate(epseq_av_all, axis=0),
        "damage": np.concatenate(damage_av_all, axis=0),
        "triaxiality": np.concatenate(sig_triax_av_all, axis=0),
        "conn_all": np.concatenate(conn_all, axis=0),
        "failed": np.concatenate(failed_all, axis=0),
        "coor": coor,
        "disp": disp,
    }


def plot_3d(plot_data, keyword, args):
    """
    Plots a 3D contour of the specified quantity on the deformed mesh.

    Args:
        plot_data (dict): Dictionary from prepare_plot_data.
        keyword (str): The quantity to plot (e.g., "stress", "strain", etc.).
        args: Object containing plotting options (e.g., args.save).
    """
    if keyword not in PLOT_CONFIG:
        print(f"Warning: Keyword '{keyword}' not supported for plotting.")
        return

    config = PLOT_CONFIG[keyword]
    
    # Unpack necessary data
    data_to_plot = plot_data[config["data_key"]]
    conn_all = plot_data["conn_all"]
    coor = plot_data["coor"]
    disp = plot_data["disp"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define faces for a hexahedral element (assuming 8-node hex)
    faces = [
        [0, 1, 2, 3], # bottom
        [4, 5, 6, 7], # top
        [0, 1, 5, 4], # front
        [1, 2, 6, 5], # right
        [2, 3, 7, 6], # back
        [3, 0, 4, 7]  # left
    ]

    # --- Color Normalization and Mapping ---
    if data_to_plot.size > 0:
        vmin = np.min(data_to_plot)
        vmax = np.max(data_to_plot)
    else:
        vmin, vmax = 0, 1 # Default range for empty data
        
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["jet"]
    edge_col='k'

    if keyword == "failed_elements":
        colors = np.array([
            [1.0, 0.0, 0.0, 1.0] if val >= 1.0 else [0.5, 0.5, 0.5, 0.02]  # RGBa
            for val in data_to_plot
        ])
        edge_col = [0.5, 0.5, 0.5, 0.05] 
    else:
        colors = cmap(norm(data_to_plot))
        

    # --- Plot Meshes ---
    for i, element in enumerate(conn_all):
        if element.size == 0:
            print(f"Skipping empty element at index {i}")
            continue  # skip empty elements
        # Plot deformed mesh with color
        verts_deformed = np.array(coor[element] + disp[element]) 
        for face in faces:
            poly_deformed = Poly3DCollection([verts_deformed[face].tolist()], 
                                             facecolors=[colors[i]], 
                                             edgecolor=edge_col, 
                                             linewidths=0.5)
            ax.add_collection3d(poly_deformed)

        # Plot undeformed mesh with dashed edges and transparent faces
        # verts_undeformed = np.array(coor[element]) 
        # for face in faces:
        #     poly_undeformed = Poly3DCollection([verts_undeformed[face].tolist()],
        #                                        facecolors=[[0, 0, 0, 0]], 
        #                                        edgecolors='k',
        #                                        linewidths=0.5,
        #                                        linestyles='dashed')
        #     ax.add_collection3d(poly_undeformed)

    # --- Colorbar ---
    if keyword != "failed_elements":
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(data_to_plot)
        # The label and format are now read from the config dictionary
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label=config["label"])
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(config["format"]))

    # --- Labels and View ---
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1]) 
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])

    # --- Optional save or show ---
    if args.save:
        fig.savefig(config["filename"])
    else:
        plt.show()
        
    plt.close(fig)

def plot_materials(coor, conn, mat, args, labels=None):
    """
    Plots the undeformed mesh with material coloring:
    - One plot with all materials together
    - One plot for each material individually

    Args:
        coor (np.ndarray): Nodal coordinates.
        conn (list of np.ndarray): List of element connectivities per material.
        mat (list): Material objects (only used for count/labels).
        args: Object with plotting options (e.g., args.save).
        labels (list of str, optional): Legend labels per material group.
    """
    cmap = plt.colormaps["tab10"]
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [1, 2, 6, 5],  # right
        [2, 3, 7, 6],  # back
        [3, 0, 4, 7],  # left
    ]

    # --- Plot all materials together ---
    fig_all = plt.figure()
    ax_all = fig_all.add_subplot(111, projection="3d")

    for i, elements in enumerate(conn):
        color = cmap(i / len(conn))
        for element in elements:
            verts = np.array(coor[element])
            for face in faces:
                poly = Poly3DCollection([verts[face].tolist()],
                                        facecolors=[color],
                                        edgecolor="k",
                                        linewidths=0.5)
                ax_all.add_collection3d(poly)

    ax_all.set_xlabel("X")
    ax_all.set_ylabel("Y")
    ax_all.set_zlabel("Z")
    ax_all.set_box_aspect([1, 1, 1])
    ax_all.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])

    if labels is None:
        labels = [f"Material {i}" for i in range(len(conn))]

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i / len(conn))) for i in range(len(conn))]
    ax_all.legend(handles, labels, loc="center left", bbox_to_anchor=(-0.2, 0.0))

    if args.save:
        fig_all.savefig("contour_all_materials.pdf")
    else:
        plt.show()
    plt.close(fig_all)

    # --- Plot each material individually ---
    for i, elements in enumerate(conn):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        color = cmap(i / len(conn))
        for element in elements:
            verts = np.array(coor[element])
            for face in faces:
                poly = Poly3DCollection([verts[face].tolist()],
                                        facecolors=[color],
                                        edgecolor="k",
                                        linewidths=0.5)
                ax.add_collection3d(poly)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])

        title = labels[i] if labels else f"Material {i}"
        ax.set_title(title)

        if args.save:
            filename = f"contour_{title.lower().replace(' ', '_')}.pdf"
            fig.savefig(filename)
        else:
            plt.show()

        plt.close(fig)

def write_XDMF(plot_data, keywords, step):
    conn_all = plot_data["conn_all"]
    coor = plot_data["coor"]
    disp = plot_data["disp"]
    elements = [("hexahedron", conn_all)]

    point_data = {}
    cell_data = {}

    for keyword in keywords:
        config = PLOT_CONFIG[keyword]
        data = plot_data[config["data_key"]]

        if config["location"] == "point":
            point_data[keyword] = data
        elif config["location"] == "element":
            cell_data[keyword] = [data]
        else:
            raise ValueError(f"Invalid location for keyword '{keyword}'")

    deformed_mesh = coor + disp

    mesh = meshio.Mesh(
        points=deformed_mesh,
        cells=elements,
        point_data={"displacement": disp, **point_data},
        cell_data=cell_data
    )

    os.makedirs("results", exist_ok=True)
    filename = os.path.join("results", f"increment_{step:04d}.xdmf")
    mesh.write(filename)