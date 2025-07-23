"""
file: parser.py
author: Philipp van der Loos

A helper function to parse gmsh files with a cohesive zone to FEM format.

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
            - 'conn_cohesive': numpy.ndarray of shape (N, 4), where N is the number of cohesive elements
                               (node IDs are 0-indexed)
            - 'conn_bulk': numpy.ndarray of shape (M, 4), where M is the number of bulk elements
                           (node IDs are 0-indexed)
            - 'nodes_by_physical_line': dict, where keys are physical line names (str) and values
                                        are numpy.ndarray of 0-indexed node IDs, sorted by X-coordinate.
                                        This includes all 1D physical lines EXCEPT the cohesive lines.
    """

    temp_nodes_dict = {}
    elements = []
    physical_names = {}
    
    bottom_line_coh_elem = [] 
    top_line_coh_elem = []

    # New dictionary to store nodes for all 1D physical lines, keyed by their name
    line_nodes_by_physical_name_temp = {}
    
    bulk_elements_connectivity_list = [] 

    current_section = None
    max_node_id = 0 

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
                pass 
            elif current_section == '$PhysicalNames':
                parts = line.split()
                if len(parts) == 3: 
                    tag_id = int(parts[1])
                    name = parts[2].strip('"')
                    physical_names[tag_id] = name
            elif current_section == '$Nodes':
                if line.isdigit(): 
                    pass
                else:
                    parts = line.split()
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    temp_nodes_dict[node_id] = np.array([x, y, z])
                    if node_id > max_node_id:
                        max_node_id = node_id
            elif current_section == '$Elements':
                if line.isdigit(): 
                    num_elements = int(line)
                else:
                    parts = list(map(int, line.split()))
                    elem_id = parts[0]
                    elem_type = parts[1]
                    num_tags = parts[2]
                    
                    phys_tag = parts[3] if num_tags >= 1 else 0

                    node_ids = parts[3 + num_tags:]
                    
                    elements.append({
                        'id': elem_id,
                        'type': elem_type,
                        'phys_tag': phys_tag,
                        'node_ids': node_ids
                    })

                    if elem_type == 1: # Gmsh element type 1 is a 2-node line
                        line_name = physical_names.get(phys_tag)
                        if line_name == bottom_line_name:
                            bottom_line_coh_elem.append({'id': elem_id, 'nodes': node_ids})
                        elif line_name == top_line_name:
                            top_line_coh_elem.append({'id': elem_id, 'nodes': node_ids})
                        elif line_name: # Store nodes for other named physical lines
                            if line_name not in line_nodes_by_physical_name_temp:
                                line_nodes_by_physical_name_temp[line_name] = []
                            line_nodes_by_physical_name_temp[line_name].extend(node_ids)
                    elif elem_type == 3: # Gmsh element type 3 is a 4-node quadrangle
                        bulk_elements_connectivity_list.append(node_ids)

    coor = np.zeros((max_node_id, 2), dtype=float) 
    for node_id, coords_xyz in temp_nodes_dict.items():
        coor[node_id - 1, 0] = coords_xyz[0] 
        coor[node_id - 1, 1] = coords_xyz[1] 

    dofs_array = np.zeros((max_node_id, 2), dtype=int)
    for node_id in range(1, max_node_id + 1):
        dofs_array[node_id - 1, 0] = (node_id - 1) * 2 
        dofs_array[node_id - 1, 1] = (node_id - 1) * 2 + 1 

    bottom_line_coh_elem.sort(key=lambda x: coor[x['nodes'][0] - 1, 0])
    top_line_coh_elem.sort(key=lambda x: coor[x['nodes'][0] - 1, 0])

    cohesive_elements_connectivity_list = [] 

    if len(bottom_line_coh_elem) != len(top_line_coh_elem):
        print(f"Warning: Mismatch in number of 1D elements found for '{bottom_line_name}' "
              f"({len(bottom_line_coh_elem)}) and '{top_line_name}' ({len(top_line_coh_elem)}). "
              "Cohesive element formation might be incomplete or incorrect.")

    for i in range(min(len(bottom_line_coh_elem), len(top_line_coh_elem))):
        bottom_elem = bottom_line_coh_elem[i]
        top_elem = top_line_coh_elem[i]

        id_bl, id_br = bottom_elem['nodes'][1], bottom_elem['nodes'][0] 
        id_t1, id_t2 = top_elem['nodes'][1], top_elem['nodes'][0] 

        coord_bl_x = coor[id_bl - 1, 0]
        coord_bl_y = coor[id_bl - 1, 1]
        coord_br_x = coor[id_br - 1, 0]
        coord_br_y = coor[id_br - 1, 1]
        coord_t1_x = coor[id_t1 - 1, 0]
        coord_t1_y = coor[id_t1 - 1, 1]
        coord_t2_x = coor[id_t2 - 1, 0]
        coord_t2_y = coor[id_t2 - 1, 1]

        if np.abs(coord_bl_x - coord_t1_x) < tolerance:
            id_tl, id_tr = id_t1, id_t2 
        elif np.abs(coord_bl_x - coord_t2_x) < tolerance: 
            id_tl, id_tr = id_t2, id_t1
        else:
            print(f"Warning: Could not find matching left node for bottom element {bottom_elem['id']} "
                  f"(node {id_bl}) with top element {top_elem['id']} (nodes {id_t1}, {id_t2}). Skipping cohesive element.")
            continue

        if np.abs(coord_br_x - coor[id_tr - 1, 0]) > tolerance:
            print(f"Warning: Right node mismatch for bottom element {bottom_elem['id']} (node {id_br}) "
                  f"and top element {top_elem['id']} (node {id_tr}). Skipping cohesive element.")
            continue
        
        if coor[id_tl - 1, 1] < np.round(coord_bl_y, 6) or coor[id_tr - 1, 1] < np.round(coord_br_y, 6):
            print(f"Warning: Top line nodes are not above bottom line nodes for elements {bottom_elem['id']} and {top_elem['id']}. Skipping cohesive element.")
            continue

        cohesive_elements_connectivity_list.append([id_bl, id_br, id_tr, id_tl])

    # Process all other 1D physical lines for output
    nodes_by_physical_line = {}
    for line_name, node_ids_list in line_nodes_by_physical_name_temp.items():
        if node_ids_list:
            unique_nodes = np.array(list(set(node_ids_list)), dtype=int) - 1 # Convert to 0-indexed and unique
            x_coords = coor[unique_nodes, 0]
            sort_indices = np.argsort(x_coords)
            nodes_by_physical_line[line_name] = unique_nodes[sort_indices]
        else:
            nodes_by_physical_line[line_name] = np.array([], dtype=int) 


    if cohesive_elements_connectivity_list:
        cohesive_elements_connectivity = np.array(cohesive_elements_connectivity_list, dtype=int) - 1
    else:
        cohesive_elements_connectivity = np.array([], dtype=int).reshape(0, 4)

    if bulk_elements_connectivity_list:
        bulk_elements_connectivity = np.array(bulk_elements_connectivity_list, dtype=int) - 1
    else:
        bulk_elements_connectivity = np.array([], dtype=int).reshape(0, 4)

    results = {
        'coor': coor,
        'dofs': dofs_array,
        'elements': elements,
        'physical_names': physical_names,
    }

    # Add cohesive and bulk connectivity
    if cohesive_elements_connectivity_list:
        results['conn_cohesive'] = np.array(cohesive_elements_connectivity_list, dtype=int) - 1
    else:
        results['conn_cohesive'] = np.array([], dtype=int).reshape(0, 4)

    if bulk_elements_connectivity_list:
        results['conn_bulk'] = np.array(bulk_elements_connectivity_list, dtype=int) - 1
    else:
        results['conn_bulk'] = np.array([], dtype=int).reshape(0, 4)

    # Process all other 1D physical lines and add them directly to the results dictionary
    for line_name, node_ids_list in line_nodes_by_physical_name_temp.items():
        if node_ids_list:
            unique_nodes = np.array(list(set(node_ids_list)), dtype=int) - 1 # Convert to 0-indexed and unique
            x_coords = coor[unique_nodes, 0]
            sort_indices = np.argsort(x_coords)
            results[line_name] = unique_nodes[sort_indices]
        else:
            results[line_name] = np.array([], dtype=int) 

    return results          