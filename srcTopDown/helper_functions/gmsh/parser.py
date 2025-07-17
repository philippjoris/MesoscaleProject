"""
file: parser.py
author: Philipp van der Loos

A helper function to parse gmsh files to FEM format.

"""

import numpy as np
import os

def parse_msh(msh_filepath, bottom_line_name="CohesiveLineBottom", top_line_name="CohesiveLineTop", tolerance=1e-3):
    """
    Parses a Gmsh MSH file (version 2.2 ASCII) to extract nodes and elements,
    identifies nodes belonging to specified physical lines, and constructs
    4-node cohesive elements.

    Args:
        msh_filepath (str): Path to the Gmsh MSH file.
        bottom_line_name (str): The physical name of the bottom cohesive line.
        top_line_name (str): The physical name of the top cohesive line.
        tolerance (float): Tolerance for matching node coordinates.

    Returns:
        dict: A dictionary containing:
            - 'coor': numpy.ndarray of shape (num_nodes, 2) storing [x, y] coordinates for each node,
                      where row index (node_id - 1) corresponds to the node_id
            - 'dofs': numpy.ndarray of shape (num_nodes, 2) storing [dof_id_x, dof_id_y] for each node
            - 'elements': [{id: int, type: int, phys_tag: int, node_ids: list}] (all elements)
            - 'physical_names': {tag_id: name}
            - 'cohesive_elements': numpy.ndarray of shape (N, 4), where N is the number of cohesive elements
            - 'bulk_elements': numpy.ndarray of shape (M, 4), where M is the number of bulk elements
    """

    # Temporary dictionary to store nodes as {node_id: [x, y, z]}, before converting to a 2D numpy array 'coor'
    temp_nodes_dict = {}
    elements = []
    physical_names = {}
    
    # Store 1D line elements specifically for the cohesive lines
    bottom_line_coh_elem = [] 
    top_line_coh_elem = []    

    # Store 1D line elements for "BottomLine" and "TopLine" physical groups
    bottom_physical_line_elem_nodes = [] 
    top_physical_line_elem_nodes = []    
    
    # Store bulk elements (4-node quads)
    bulk_elements_connectivity_list = [] 

    current_section = None
    max_node_id = 0 # To determine the size of the 'coor' and 'dofs' arrays

    with open(msh_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('$'):
                current_section = line
                continue
            elif line.startswith('$End'):
                current_section = None
                continue

            if current_section == '$MeshFormat':
                # Read mesh format info (e.g., 2.2 0 8)
                pass # Not strictly needed for this parser, but good to acknowledge
            elif current_section == '$PhysicalNames':
                parts = line.split()
                if len(parts) == 3: # Format: dim tag "name"
                    tag_id = int(parts[1])
                    name = parts[2].strip('"')
                    physical_names[tag_id] = name
            elif current_section == '$Nodes':
                if line.isdigit(): # First line in $Nodes is number of nodes
                    pass
                else:
                    parts = line.split()
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    # Store all 3 coordinates temporarily, we'll pick X,Y later for 'coor'
                    temp_nodes_dict[node_id] = np.array([x, y, z])
                    if node_id > max_node_id:
                        max_node_id = node_id
            elif current_section == '$Elements':
                if line.isdigit(): # First line in $Elements is number of elements
                    num_elements = int(line)
                else:
                    parts = list(map(int, line.split()))
                    elem_id = parts[0]
                    elem_type = parts[1]
                    num_tags = parts[2]
                    
                    # Default Gmsh tags: physical_tag, elementary_tag, partition_tag(s)
                    phys_tag = parts[3] if num_tags >= 1 else 0

                    # Node connectivity starts after tags
                    node_ids = parts[3 + num_tags:]
                    
                    elements.append({
                        'id': elem_id,
                        'type': elem_type,
                        'phys_tag': phys_tag,
                        'node_ids': node_ids
                    })

                    # If it's a 1D line element (type 1) and belongs to our cohesive lines
                    if elem_type == 1: # Gmsh element type 1 is a 2-node line
                        if physical_names.get(phys_tag) == bottom_line_name:
                            bottom_line_coh_elem.append({'id': elem_id, 'nodes': node_ids})
                        elif physical_names.get(phys_tag) == top_line_name:
                            top_line_coh_elem.append({'id': elem_id, 'nodes': node_ids})
                        elif physical_names.get(phys_tag) == "BottomLine":
                            bottom_physical_line_elem_nodes.append(node_ids)
                        elif physical_names.get(phys_tag) == "TopLine":
                            top_physical_line_elem_nodes.append(node_ids)                            
                    elif elem_type == 3: # Gmsh element type 3 is a 4-node quadrangle
                        # Append only the node IDs for bulk elements
                        bulk_elements_connectivity_list.append(node_ids)

    # Create the 'coor' array: a 2D NumPy array of [x, y] coordinates per node
    coor = np.zeros((max_node_id, 2), dtype=float) # Shape changed to (max_node_id, 2)
    for node_id, coords_xyz in temp_nodes_dict.items():
        # Store X and Y coordinates at coor[node_id - 1]
        coor[node_id - 1, 0] = coords_xyz[0] # X coordinate
        coor[node_id - 1, 1] = coords_xyz[1] # Y coordinate

    # Create the 'dofs' array: each row is [dof_id_x, dof_id_y] for the corresponding node (0-indexed)
    dofs_array = np.zeros((max_node_id, 2), dtype=int)
    for node_id in range(1, max_node_id + 1):
        dofs_array[node_id - 1, 0] = (node_id - 1) * 2     # x_dof_id
        dofs_array[node_id - 1, 1] = (node_id - 1) * 2 + 1 # y_dof_id

    # --- Step 3: Sort 1D Line Elements by their first node's X-coordinate ---
    # Access X-coordinate from the 'coor' array: coor[node_id - 1, 0]
    bottom_line_coh_elem.sort(key=lambda x: coor[x['nodes'][0] - 1, 0])
    top_line_coh_elem.sort(key=lambda x: coor[x['nodes'][0] - 1, 0])

    # --- Step 4: Form 4-node Cohesive Elements ---
    cohesive_elements_connectivity_list = [] # Temporary list to store connectivity before converting to numpy array

    # We expect the number of 1D elements on both lines to be the same
    if len(bottom_line_coh_elem) != len(top_line_coh_elem):
        print(f"Warning: Mismatch in number of 1D elements found for '{bottom_line_name}' "
              f"({len(bottom_line_coh_elem)}) and '{top_line_name}' ({len(top_line_coh_elem)}). "
              "Cohesive element formation might be incomplete or incorrect.")

    # Iterate through the sorted bottom line elements and find corresponding top line elements
    for i in range(min(len(bottom_line_coh_elem), len(top_line_coh_elem))):
        bottom_elem = bottom_line_coh_elem[i]
        top_elem = top_line_coh_elem[i]

        # Get node IDs for current bottom and top 1D elements
        id_bl, id_br = bottom_elem['nodes'][1], bottom_elem['nodes'][0] # lower_left, lower_right
        id_t1, id_t2 = top_elem['nodes'][1], top_elem['nodes'][0]       # top_node1, top_node2

        # Get coordinates of these nodes from the 'coor' 2D NumPy array
        coord_bl_x = coor[id_bl - 1, 0]
        coord_bl_y = coor[id_bl - 1, 1]
        coord_br_x = coor[id_br - 1, 0]
        coord_br_y = coor[id_br - 1, 1]
        coord_t1_x = coor[id_t1 - 1, 0]
        coord_t1_y = coor[id_t1 - 1, 1]
        coord_t2_x = coor[id_t2 - 1, 0]
        coord_t2_y = coor[id_t2 - 1, 1]

        # Determine the correct order for top nodes (upper_left, upper_right)
        # Compare X-coordinates for alignment
        if np.abs(coord_bl_x - coord_t1_x) < tolerance:
            id_tl, id_tr = id_t1, id_t2 # top_left, top_right
        elif np.abs(coord_bl_x - coord_t2_x) < tolerance: # Top element might be reversed
            id_tl, id_tr = id_t2, id_t1
        else:
            print(f"Warning: Could not find matching left node for bottom element {bottom_elem['id']} "
                  f"(node {id_bl}) with top element {top_elem['id']} (nodes {id_t1}, {id_t2}). Skipping cohesive element.")
            continue

        # Final check for right node alignment (X-coordinate)
        if np.abs(coord_br_x - coor[id_tr - 1, 0]) > tolerance:
             print(f"Warning: Right node mismatch for bottom element {bottom_elem['id']} (node {id_br}) "
                   f"and top element {top_elem['id']} (node {id_tr}). Skipping cohesive element.")
             continue
        
        # Ensure top nodes are indeed 'above' bottom nodes (Y-coordinate check)
        if coor[id_tl - 1, 1] < np.round(coord_bl_y, 6) or coor[id_tr - 1, 1] < np.round(coord_br_y, 6):
            print(coor[id_tl - 1, 1], np.round(coord_bl_y, 6))
            print(f"Warning: Top line nodes are not above bottom line nodes for elements {bottom_elem['id']} and {top_elem['id']}. Skipping cohesive element.")
            continue

        # Form the cohesive element: (lower_left, lower_right, upper_right, upper_left)
        cohesive_elements_connectivity_list.append([id_bl, id_br, id_tr, id_tl]) # Append just the list of node IDs

    # Collect unique node IDs from "BottomLine" elements
    unique_nodes_bottom_line = set()
    for node_pair in bottom_physical_line_elem_nodes:
        unique_nodes_bottom_line.update(node_pair)
    
    if unique_nodes_bottom_line:
        nodes_bottom_line = np.array(list(unique_nodes_bottom_line), dtype=int) - 1
        x_coords_bottom = coor[nodes_bottom_line, 0]
        sort_indices_bottom = np.argsort(x_coords_bottom)
        nodes_bottom_line = nodes_bottom_line[sort_indices_bottom]
    else:
        nodes_bottom_line = np.array([], dtype=int) 

    unique_nodes_top_line = set()
    for node_pair in top_physical_line_elem_nodes:
        unique_nodes_top_line.update(node_pair)

    if unique_nodes_top_line:
        nodes_top_line = np.array(list(unique_nodes_top_line), dtype=int) - 1
        x_coords_top = coor[nodes_top_line, 0]
        sort_indices_top = np.argsort(x_coords_top)
        nodes_top_line = nodes_top_line[sort_indices_top]
    else:
        nodes_top_line = np.array([], dtype=int) 

    # Convert connectivity lists to NumPy arrays, ensuring they are always initialized
    if cohesive_elements_connectivity_list:
        cohesive_elements_connectivity = np.array(cohesive_elements_connectivity_list, dtype=int) - 1
    else:
        cohesive_elements_connectivity = np.array([], dtype=int).reshape(0, 4) # Explicitly create empty (0,4) array

    if bulk_elements_connectivity_list:
        bulk_elements_connectivity = np.array(bulk_elements_connectivity_list, dtype=int) - 1
    else:
        bulk_elements_connectivity = np.array([], dtype=int).reshape(0, 4) # Explicitly create empty (0,4) array

    return {
        'coor': coor, # Return the 2D NumPy array for coordinates
        'dofs': dofs_array, # Return the NumPy array for DOFs
        'elements': elements,
        'physical_names': physical_names,
        'conn_cohesive': cohesive_elements_connectivity, 
        'conn_bulk': bulk_elements_connectivity, 
        'nodesTopEdge': nodes_top_line,
        'nodesBottomEdge': nodes_bottom_line
    }

# --- Example Usage (replace 'your_mesh_file.msh' with your actual file path) ---
# if __name__ == "__main__":
#     # Create a dummy MSH file for testing the parser
#    # In a real scenario, you would generate this with Gmsh
#    dummy_msh_content = """
# $MeshFormat
# 2.2 0 8
# $EndMeshFormat
# $PhysicalNames
# 5
# 1 1 "MainDomain_Bottom"
# 1 2 "MainDomain_Top"
# 1 3 "CohesiveLineBottom"
# 1 4 "CohesiveLineTop"
# 2 5 "MainDomain_Bulk"
# $EndPhysicalNames
# $Nodes
# 10
# 1 0.0 0.0 0.0
# 2 1.0 0.0 0.0
# 3 0.0 0.3 0.0
# 4 1.0 0.3 0.0
# 5 2.0 0.0 0.0
# 6 3.0 0.0 0.0
# 7 2.0 0.3 0.0
# 8 3.0 0.3 0.0
# 9 0.0 0.6 0.0
# 10 1.0 0.6 0.0
# $EndNodes
# $Elements
# 6
# 1 1 1 3 1 2
# 2 1 1 3 5 6
# 3 1 1 4 3 4
# 4 1 1 4 7 8
# 5 3 2 5 1 1 2 4 3
# 6 3 2 5 1 2 5 7 4
# $EndElements
# """
#     input_mesh = "surface_w_hole_cohesive.msh"
# 
#     #with open("dummy_mesh.msh", "w") as f:
#     #    f.write(dummy_msh_content)
#     current_directory = os.path.dirname(os.path.abspath(__file__))
#     mesh_file_path = os.path.join(current_directory, input_mesh)
# 
#     msh_data = parse_msh_and_create_cohesive_elements(mesh_file_path)
# 
#     print("--- Parsed Coordinates (coor) ---")
#     print(f"Type: {type(msh_data['coor'])}")
#     print(msh_data['coor'])
# 
#     print("\n--- Parsed DOFs (Node ID to DOF IDs) ---")
#     print(f"Type: {type(msh_data['dofs'])}")
#     print(msh_data['dofs'])
# 
#     print("\n--- Parsed Elements (All) ---")
#     for elem in msh_data['elements']:
#         print(f"Element {elem['id']}: Type {elem['type']}, PhysTag {elem['phys_tag']} ({msh_data['physical_names'].get(elem['phys_tag'], 'N/A')}), Nodes {elem['node_ids']}")
# 
#     print("\n--- Identified Cohesive Elements (Connectivity Only) ---")
#     if msh_data['cohesive_elements'].size > 0: # Check if array is not empty
#         print(f"Type: {type(msh_data['cohesive_elements'])}")
#         print(msh_data['cohesive_elements'])
#     else:
#         print("No cohesive elements found based on the specified physical line names.")
# 
#     print("\n--- Identified Bulk Elements (Connectivity Only) ---")
#     if msh_data['bulk_elements'].size > 0: # Check if array is not empty
#         print(f"Type: {type(msh_data['bulk_elements'])}")
#         print(msh_data['bulk_elements'])
#     else:
#         print("No bulk (4-node quad) elements found.")
# 
#     # Clean up dummy file
#     # os.remove("dummy_mesh.msh")
