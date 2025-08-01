"""
file: parser.py
author: Philipp van der Loos

A helper function to parse gmsh files with a cohesive zone to FEM format.

"""

import numpy as np
import os

def parse_msh(msh_filepath, cohesive_line_name="Cohesive", tolerance=1e-3):
    """
    Parses a Gmsh MSH file (version 2.2 ASCII) to extract nodes and elements,
    identifies nodes belonging to a specified cohesive physical line,
    duplicates these nodes to create a cohesive zone, and constructs
    4-node cohesive elements and updates bulk element connectivity.

    Args:
        msh_filepath (str): Path to the Gmsh MSH file.
        cohesive_line_name (str): The physical name of the single cohesive line.
        tolerance (float): Tolerance for matching node coordinates.

    Returns:
        dict: A dictionary containing:
            - 'coor': numpy.ndarray of shape (num_nodes, 2) storing [x, y] coordinates for each node,
                      where row index (node_id - 1) corresponds to the node_id
            - 'dofs': numpy.ndarray of shape (num_nodes, 2) storing [dof_id_x, dof_id_y] for each node
            - 'elements': [{id: int, type: int, phys_tag: int, node_ids: list}] (all elements, with updated node IDs)
            - 'physical_names': {tag_id: name}
            - 'conn_cohesive': numpy.ndarray of shape (N, 4), where N is the number of cohesive elements
                               (node IDs are 0-indexed)
            - 'conn_bulk': numpy.ndarray of shape (M, 4), where M is the number of bulk elements
                           (node IDs are 0-indexed)
            - Other physical line names as keys, with values being numpy.ndarray of 0-indexed node IDs,
              sorted by X-coordinate. These include all 1D physical lines EXCEPT the cohesive line.
    """

    temp_nodes_dict = {}
    elements = []
    physical_names = {}
    
    cohesive_line_elements = [] 
    bulk_elements_connectivity_list = []

    # Dictionary to store nodes for all 1D physical lines (excluding the cohesive line), keyed by their name
    line_nodes_by_physical_name_temp = {}
    
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
                    num_elements = int(line) # This line is often the count, not an element
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
                        'node_ids': node_ids # Storing original node IDs for now
                    })

                    if elem_type == 1: # Gmsh element type 1 is a 2-node line
                        line_name = physical_names.get(phys_tag)
                        if line_name == cohesive_line_name:
                            cohesive_line_elements.append({'id': elem_id, 'nodes': node_ids})
                        elif line_name: # Store nodes for other named physical lines
                            if line_name not in line_nodes_by_physical_name_temp:
                                line_nodes_by_physical_name_temp[line_name] = []
                            line_nodes_by_physical_name_temp[line_name].extend(node_ids)
                    elif elem_type == 3: # Gmsh element type 3 is a 4-node quadrangle
                        bulk_elements_connectivity_list.append(node_ids)

    # Convert nodes dictionary to a numpy array for coordinates
    # Initialize coor with enough space, anticipating new nodes
    initial_num_nodes = max_node_id
    coor = np.zeros((initial_num_nodes, 2), dtype=float) 
    for node_id, coords_xyz in temp_nodes_dict.items():
        coor[node_id - 1, 0] = coords_xyz[0] 
        coor[node_id - 1, 1] = coords_xyz[1] 

    # Sort the cohesive line elements based on the X-coordinate of their first node
    # This ensures we process them in order along the line
    cohesive_line_elements.sort(key=lambda x: coor[x['nodes'][0] - 1, 0])

    # Map for original node IDs on the cohesive line to their new duplicated node IDs
    original_to_duplicated_node_map = {}
    
    # List to store new cohesive element connectivities
    cohesive_elements_connectivity_list = []
    
    # Store coordinates of duplicated nodes temporarily
    duplicated_node_coords = []

    # Create duplicated nodes and cohesive elements
    next_new_node_id = max_node_id + 1

    # Get unique nodes on the cohesive line, sorted by X-coordinate
    unique_cohesive_line_node_ids = []
    for elem in cohesive_line_elements:
        unique_cohesive_line_node_ids.extend(elem['nodes'])
    unique_cohesive_line_node_ids = sorted(list(set(unique_cohesive_line_node_ids)), 
                                           key=lambda node_id: coor[node_id - 1, 0])

    # For each unique node on the cohesive line, create a duplicated node
    # The duplicated node will be slightly offset (e.g., in Y-direction) if needed for visualization,
    # but for FEM, it's typically at the same location initially.
    for original_node_id in unique_cohesive_line_node_ids:
        if original_node_id not in original_to_duplicated_node_map:
            original_to_duplicated_node_map[original_node_id] = next_new_node_id
            duplicated_node_coords.append(coor[original_node_id - 1]) # Duplicated node has same coords
            next_new_node_id += 1

    # Now, reconstruct `coor` with the duplicated nodes
    num_original_nodes = coor.shape[0]
    num_duplicated_nodes = len(duplicated_node_coords)
    new_coor = np.zeros((num_original_nodes + num_duplicated_nodes, 2), dtype=float)
    new_coor[:num_original_nodes, :] = coor

    for i, coords in enumerate(duplicated_node_coords):
        new_coor[num_original_nodes + i, :] = coords
    coor = new_coor
    max_node_id = coor.shape[0]

    # Create cohesive elements based on the sorted cohesive line elements
    for i in range(len(cohesive_line_elements)):
        elem = cohesive_line_elements[i]
        # Nodes of the 1D cohesive line element (original IDs)
        node_id_1, node_id_2 = sorted(elem['nodes'], key=lambda n_id: coor[n_id - 1, 0])

        # Get the duplicated nodes corresponding to node_id_1 and node_id_2
        # The first original node forms the bottom-left of the cohesive element
        # The second original node forms the bottom-right of the cohesive element
        # The duplicated node of the first forms the top-left
        # The duplicated node of the second forms the top-right
        id_bl = node_id_1
        id_br = node_id_2
        id_tl = original_to_duplicated_node_map[node_id_1]
        id_tr = original_to_duplicated_node_map[node_id_2]
        
        cohesive_elements_connectivity_list.append([id_bl, id_br, id_tr, id_tl])

    # Update bulk element connectivity: replace original cohesive line node IDs with duplicated ones
    # for the side that touches the cohesive zone.
    # This assumes the cohesive line is "on top" of the bulk elements for 2D (positive Y direction).
    # You might need to adjust this logic based on your mesh generation strategy
    # (e.g., if the cohesive line is at the bottom of the bulk elements).
    updated_bulk_elements_connectivity_list = []
    for bulk_elem_nodes in bulk_elements_connectivity_list:
        updated_nodes = list(bulk_elem_nodes)
        # Assuming 4-node quadrilaterals: Gmsh node order for quad is usually BL, BR, TR, TL
        # If the bottom edge of the bulk element is adjacent to the cohesive line,
        # then the bottom nodes (0-indexed) 0 and 1 of the bulk element would be
        # replaced with their duplicated counterparts.
        # This requires identifying which nodes of the bulk element are on the cohesive line.
        
        # A more robust approach: check if any node of the bulk element is an original cohesive line node
        # If it is, and its Y-coordinate is such that it's on the "top" of the bulk element (facing cohesive zone),
        # replace it with its duplicated counterpart.
        
        # Get the coordinates of the bulk element nodes
        node_coords = [temp_nodes_dict[node_id] for node_id in bulk_elem_nodes]
        y_coords = [coords[1] for coords in node_coords]
        
        # Find the maximum Y-coordinate among the bulk element's nodes
        max_y_bulk_elem = max(y_coords)

        for i, node_id in enumerate(bulk_elem_nodes):
            if node_id in original_to_duplicated_node_map:
                # Check if this node is on the "top" edge of the bulk element facing the cohesive zone.
                # This is a simplification; more complex geometry might require checking adjacency.
                # For a simple rectangular mesh with cohesive zone at the top,
                # nodes with y_coord close to max_y_bulk_elem are likely the top nodes.
                if np.abs(coor[node_id - 1, 1] - max_y_bulk_elem) < tolerance:
                    updated_nodes[i] = original_to_duplicated_node_map[node_id]
        
        updated_bulk_elements_connectivity_list.append(updated_nodes)


    # Update the 'elements' list with new node IDs
    for elem in elements:
        if elem['type'] == 3: # Only update 4-node quadrangles (bulk elements)
            original_node_ids = elem['node_ids']
            updated_elem_nodes = []
            
            # Find the physical tag of the bulk element to infer its position relative to the cohesive line
            bulk_phys_name = physical_names.get(elem['phys_tag'])

            # Determine the original nodes that form the 'top' edge of the bulk element
            # This is critical and depends on how your mesh is generated.
            # Assuming a standard Gmsh quad (node_ids: bottom-left, bottom-right, top-right, top-left)
            # and that the cohesive line is placed directly on top of these bulk elements.
            # So, the 'top-right' and 'top-left' nodes of the bulk element will be reassigned.
            
            # Identify the top two nodes of the bulk element by their Y-coordinates
            current_elem_nodes_coords = [temp_nodes_dict[nid] for nid in original_node_ids]
            current_elem_nodes_y_coords = [coord[1] for coord in current_elem_nodes_coords]
            
            # Find the two nodes with the highest Y-coordinates (these are assumed to be the top nodes)
            # A more robust approach might involve checking edge connectivity or physical groups
            y_sorted_indices = np.argsort(current_elem_nodes_y_coords)
            top_node_original_ids = sorted([original_node_ids[y_sorted_indices[-1]], original_node_ids[y_sorted_indices[-2]]])
            
            for original_node_id in original_node_ids:
                if original_node_id in top_node_original_ids and original_node_id in original_to_duplicated_node_map:
                    updated_elem_nodes.append(original_to_duplicated_node_map[original_node_id])
                else:
                    updated_elem_nodes.append(original_node_id)
            elem['node_ids'] = updated_elem_nodes

    # Finally, construct the dofs array for the new total number of nodes
    dofs_array = np.zeros((max_node_id, 2), dtype=int)
    for node_id in range(1, max_node_id + 1):
        dofs_array[node_id - 1, 0] = (node_id - 1) * 2 
        dofs_array[node_id - 1, 1] = (node_id - 1) * 2 + 1 

    # Process all other 1D physical lines for output
    # These also need their node IDs converted to 0-indexed
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
        cohesive_elements_connectivity = np.array(cohesive_elements_connectivity_list, dtype=int) - 1 # Convert to 0-indexed
    else:
        cohesive_elements_connectivity = np.array([], dtype=int).reshape(0, 4)

    if updated_bulk_elements_connectivity_list:
        bulk_elements_connectivity = np.array(updated_bulk_elements_connectivity_list, dtype=int) - 1 # Convert to 0-indexed
    else:
        bulk_elements_connectivity = np.array([], dtype=int).reshape(0, 4)

    results = {
        'coor': coor,
        'dofs': dofs_array,
        'elements': elements, # This list now contains updated node IDs for bulk elements
        'physical_names': physical_names,
        'conn_cohesive': cohesive_elements_connectivity,
        'conn_bulk': bulk_elements_connectivity,
    }

    # Add other 1D physical lines directly to the results dictionary
    for line_name, nodes_array in nodes_by_physical_line.items():
        results[line_name] = nodes_array

    return results