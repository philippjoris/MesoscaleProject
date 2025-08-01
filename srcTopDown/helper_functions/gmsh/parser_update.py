"""
file: parser.py
author: Philipp van der Loos

A helper function to parse gmsh files with a cohesive zone to FEM format.

"""

import numpy as np
import collections

def parse_msh(msh_filepath, cohesive_line_names=["CohesiveInterface", "CohesiveParticle"], tolerance=1e-3):
    """
    Parses a Gmsh MSH file (version 2.2 ASCII) to extract nodes and elements,
    identifies nodes belonging to specified cohesive physical lines,
    duplicates these nodes based on the *specific interface* they belong to,
    and constructs 4-node cohesive elements and updates bulk element connectivity.

    Args:
        msh_filepath (str): Path to the Gmsh MSH file.
        cohesive_line_names (list): A list of physical names of the cohesive lines.
        tolerance (float): Tolerance for matching node coordinates.

    Returns:
        dict: A dictionary containing:
            - 'coor': numpy.ndarray of shape (num_nodes, 2) storing [x, y] coordinates for each node,
                      where row index (node_id - 1) corresponds to the node_id
            - 'dofs': numpy.ndarray of shape (num_nodes, 2) storing [dof_id_x, dof_id_y] for each node
            - 'elements': [{id: int, type: int, phys_tag: int, node_ids: list}] (all elements, with updated node IDs)
            - 'physical_names': {tag_id: name}
            - 'conn_CohesiveInterface': numpy.ndarray of shape (N, 4), where N is the number of cohesive elements
                                        for CohesiveInterface (node IDs are 0-indexed)
            - 'conn_CohesiveParticle': numpy.ndarray of shape (M, 4), where M is the number of cohesive elements
                                        for CohesiveParticle (node IDs are 0-indexed)
            - 'conn_bulk': numpy.ndarray of shape (P, 4), where P is the number of bulk elements
                            (node IDs are 0-indexed)
            - Other physical line names as keys, with values being numpy.ndarray of 0-indexed node IDs,
              sorted by X-coordinate or traversed order as appropriate.
              These include all 1D physical lines EXCEPT the cohesive lines.
    """

    temp_nodes_dict = {}
    elements_raw = [] # Store elements as they are parsed, with original node IDs
    physical_names = {}
    cohesive_line_elements_by_name = {name: [] for name in cohesive_line_names}
    line_nodes_by_physical_name_temp = {}
    max_node_id = 0
    
    # Track physical tags for quick lookup
    cohesive_tags = set()

    current_section = None
    with open(msh_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('$'):
                current_section = line
                continue
            elif line.startswith('$End'):
                current_section = None
                continue

            if current_section == '$PhysicalNames':
                parts = line.split()
                if len(parts) == 3:
                    tag_id = int(parts[1])
                    name = parts[2].strip('"')
                    physical_names[tag_id] = name
                    # If it's a cohesive line, store its tag
                    if name in cohesive_line_names:
                        cohesive_tags.add(tag_id)
            elif current_section == '$Nodes':
                if line.isdigit(): # Skip the count line
                    pass
                else:
                    parts = line.split()
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    temp_nodes_dict[node_id] = np.array([x, y, z])
                    if node_id > max_node_id: max_node_id = node_id
            elif current_section == '$Elements':
                if line.isdigit(): # Skip the count line
                    pass
                else:
                    parts = list(map(int, line.split()))
                    elem_id = parts[0]
                    elem_type = parts[1]
                    num_tags = parts[2]
                    phys_tag = parts[3] if num_tags >= 1 else 0
                    node_ids = parts[3 + num_tags:]

                    elements_raw.append({ # Store raw elements with original node IDs and phys_tag
                        'id': elem_id,
                        'type': elem_type,
                        'phys_tag': phys_tag,
                        'node_ids': node_ids
                    })

                    if elem_type == 1: # Gmsh element type 1 (2-node line)
                        line_name = physical_names.get(phys_tag)
                        if line_name in cohesive_line_names:
                            cohesive_line_elements_by_name[line_name].append({'id': elem_id, 'nodes': node_ids, 'phys_tag': phys_tag})
                        elif line_name:
                            if line_name not in line_nodes_by_physical_name_temp:
                                line_nodes_by_physical_name_temp[line_name] = []
                            line_nodes_by_physical_name_temp[line_name].extend(node_ids)

    # Convert original nodes dictionary to numpy array
    coor = np.zeros((max_node_id, 2), dtype=float)
    for node_id, coords_xyz in temp_nodes_dict.items():
        coor[node_id - 1, 0] = coords_xyz[0]
        coor[node_id - 1, 1] = coords_xyz[1]

    # This will store {node_id: set_of_physical_line_tags_it_belongs_to}
    # With this step we know to which physical domains each node belongs to.
    node_to_physical_line_tags = {}
    for elem_data in elements_raw:
        if elem_data['type'] == 1: # It's a 1D line element
            for node_id in elem_data['node_ids']:
                if node_id not in node_to_physical_line_tags:
                    node_to_physical_line_tags[node_id] = set()
                node_to_physical_line_tags[node_id].add(elem_data['phys_tag'])
    
    # Map: {original_node_id: {interface_name: duplicated_node_id}}
    # The original node_id will serve as the "bottom" or "inner" side.
    # This dublicated node will serve as the "top" or "outer" side.    
    interface_node_duplicates = {name: {} for name in cohesive_line_names}

    next_available_node_id = max_node_id + 1
    new_coords_list = []

    # Iterate through all 1D cohesive line elements to identify nodes for duplication
    for interface_name in cohesive_line_names:
        for line_elem in cohesive_line_elements_by_name[interface_name]:
            for original_node_id in line_elem['nodes']:
                if original_node_id not in interface_node_duplicates[interface_name]:
                    interface_node_duplicates[interface_name][original_node_id] = next_available_node_id
                    new_coords_list.append(coor[original_node_id - 1]) # Duplicated node has same coords
                    next_available_node_id += 1
    
    # Extend the coordinate array with all new duplicated nodes
    num_original_nodes = coor.shape[0]
    total_new_nodes = len(new_coords_list)
    final_coor = np.zeros((num_original_nodes + total_new_nodes, 2), dtype=float)
    final_coor[:num_original_nodes, :] = coor # Copy original nodes

    for i, coords in enumerate(new_coords_list):
        final_coor[num_original_nodes + i, :] = coords
    coor = final_coor # Update global coordinate array
    max_node_id = coor.shape[0] # Update total number of nodes

    # --- Construct Cohesive Elements Connectivity ---
    cohesive_elements_connectivity_list_by_name = {name: [] for name in cohesive_line_names}

    for interface_name in cohesive_line_names:
        current_cohesive_elements_conn = cohesive_elements_connectivity_list_by_name[interface_name]
        current_node_map = interface_node_duplicates[interface_name]

        if interface_name == "CohesiveInterface":
            # Sort cohesive line elements by the x-coordinate of their first node for consistent local connectivity (left to right)
            cohesive_line_elements_by_name[interface_name].sort(key=lambda x: coor[x['nodes'][0] - 1, 0] if x['nodes'] else float('inf'))

            for line_elem in cohesive_line_elements_by_name[interface_name]:
                # Sort by X to get (left, right) for linear interfaces
                n1_orig, n2_orig = sorted(line_elem['nodes'], key=lambda n_id: coor[n_id - 1, 0]) 

                id_bl = n1_orig
                id_br = n2_orig
                id_tl = current_node_map[n1_orig]
                id_tr = current_node_map[n2_orig]

                current_cohesive_elements_conn.append([id_bl, id_br, id_tr, id_tl]) # Node IDs are 1-indexed at this stage

        elif interface_name == "CohesiveParticle":
            # Graph traversal for "CohesiveParticle" to order elements along the contour
            
            # 1. Build a map of nodes to their cohesive line elements for *this* particle interface
            particle_node_to_elements = collections.defaultdict(list)
            for line_elem in cohesive_line_elements_by_name[interface_name]:
                for node_id in line_elem['nodes']:
                    particle_node_to_elements[node_id].append(line_elem)

            all_particle_elements = list(cohesive_line_elements_by_name[interface_name])
            processed_element_ids = set()
            
            # This list will store ordered sequences for potentially multiple disconnected particles
            all_ordered_particle_contours = []

            # Loop to find all distinct particle contours/lines
            while len(processed_element_ids) < len(all_particle_elements):
                # Find an un-processed element to start a new contour
                start_elem = None
                for elem in all_particle_elements:
                    if elem['id'] not in processed_element_ids:
                        start_elem = elem
                        break
                
                if start_elem is None: # No more un-processed elements
                    break # All elements have been processed

                # Pick a starting node for this contour.
                # For a closed loop, any node works, but for consistency, pick one
                # For an open line, pick an end node (node connected to only one element of this type)
                
                # Heuristic: Find a node that is an 'end' node (degree 1 in the cohesive graph)
                # or if a closed loop, pick the first node of the start_elem.
                start_node_for_contour = None
                for node_id in start_elem['nodes']:
                    if len(particle_node_to_elements[node_id]) == 1: # This is an end node
                        start_node_for_contour = node_id
                        break
                if start_node_for_contour is None: # It's a closed loop or all nodes have degree > 1
                    # Choose the node with minimum X, then minimum Y as a consistent start for closed loops
                    # within the nodes of the current start_elem.
                    start_node_for_contour = sorted(start_elem['nodes'], key=lambda n_id: (coor[n_id - 1, 0], coor[n_id - 1, 1]))[0]

                current_node_id = start_node_for_contour
                previous_node_id = None # Used to track the direction of traversal

                current_contour_ordered_elements = []

                # Traverse the current contour
                while True:
                    next_elem_found = False
                    
                    # Find potential next elements from current_node_id
                    candidate_elements = [e for e in particle_node_to_elements[current_node_id] if e['id'] not in processed_element_ids]

                    if not candidate_elements: # No un-processed elements connected to current_node
                        break # End of this contour

                    # Logic to select the 'next' element and orient nodes
                    if len(candidate_elements) == 1:
                        # Simple case: only one path forward
                        next_elem = candidate_elements[0]
                    else:
                        # More complex: current_node_id is part of a bifurcation or a return point.
                        # This typically happens only if `previous_node_id` is None (first element)
                        # or if there's an actual mesh error/discontinuity.
                        # For a well-formed line, there should usually be only one unvisited next element.
                        # We pick one that doesn't immediately go back to the previous node.
                        next_elem = None
                        for cand_elem in candidate_elements:
                            other_node_in_cand = next(n for n in cand_elem['nodes'] if n != current_node_id)
                            if other_node_in_cand != previous_node_id:
                                next_elem = cand_elem
                                break
                        if next_elem is None: # Should not happen for valid single contours
                            print(f"Warning: Stalled traversal at node {current_node_id}. Possible complex geometry or mesh error.")
                            break

                    current_contour_ordered_elements.append(next_elem)
                    processed_element_ids.add(next_elem['id'])

                    # Determine the next current_node_id for the traversal
                    old_current_node_id = current_node_id
                    current_node_id = next(n for n in next_elem['nodes'] if n != current_node_id)
                    previous_node_id = old_current_node_id
                    next_elem_found = True

                if current_contour_ordered_elements:
                    all_ordered_particle_contours.append(current_contour_ordered_elements)

            # Now, use the ordered contours to build the cohesive element connectivity
            for contour_elements in all_ordered_particle_contours:
                for i, line_elem in enumerate(contour_elements):
                    # Determine n1_orig (bottom-left) and n2_orig (bottom-right)
                    # The order of nodes in the 'line_elem' from Gmsh is usually consistent along the curve.
                    # We just need to make sure 'n1_orig' is the node we entered the element from
                    # and 'n2_orig' is the node we're leaving to for consistent element definition.

                    if i == 0: # First element in the contour
                        # For the very first element of a contour, we need to decide its 'start' and 'end' node.
                        # We make 'n1_orig' the node closest to the start_node_for_contour that initiated this specific contour.
                        n1_candidate, n2_candidate = line_elem['nodes']
                        if n1_candidate == start_node_for_contour:
                             n1_orig, n2_orig = n1_candidate, n2_candidate
                        else:
                             n1_orig, n2_orig = n2_candidate, n1_candidate
                    else:
                        # For subsequent elements, ensure the order is continuous from the previous element.
                        prev_elem = contour_elements[i-1]
                        # The 'second' node of the previous element should be the 'first' node of the current element.
                        expected_n1 = next(n for n in prev_elem['nodes'] if n != previous_node_id) # The node that was current_node_id
                        
                        if line_elem['nodes'][0] == expected_n1:
                            n1_orig, n2_orig = line_elem['nodes'][0], line_elem['nodes'][1]
                        else:
                            n1_orig, n2_orig = line_elem['nodes'][1], line_elem['nodes'][0]

                    id_bl = n1_orig
                    id_br = n2_orig
                    id_tl = current_node_map[n1_orig]
                    id_tr = current_node_map[n2_orig]

                    current_cohesive_elements_conn.append([id_bl, id_br, id_tr, id_tl]) # Node IDs are 1-indexed


    # --- Update Bulk Elements Connectivity ---
    final_bulk_elements_connectivity = []
    elements_final = [] # List to store all elements with updated node IDs

    # Get physical tags for quick lookup
    cohesive_interface_tag = next( (tag for tag, name in physical_names.items() if name == "CohesiveInterface"), None)
    cohesive_particle_tag = next( (tag for tag, name in physical_names.items() if name == "CohesiveParticle"), None)
    upper_domain_tag = next( (tag for tag, name in physical_names.items() if name == "UpperDomain"), None)
    lower_domain_tag = next( (tag for tag, name in physical_names.items() if name == "LowerDomain"), None)
    particle_domain_tag = next( (tag for tag, name in physical_names.items() if name == "ParticleDomain"), None)


    for elem_data in elements_raw: # Iterate through original parsed elements
        if elem_data['type'] == 3: # This is a 4-node quad (bulk element)
            original_node_ids = elem_data['node_ids']
            updated_node_ids = list(original_node_ids) # Start with original nodes
            bulk_phys_tag = elem_data['phys_tag']

            for i, original_node_id in enumerate(original_node_ids):
                # Get the set of physical line tags this node belongs to
                node_line_tags = node_to_physical_line_tags.get(original_node_id, set())

                # Logic for mapping node based on which bulk domain it's in and which interface it touches

                # Case 1: Bulk element is UpperDomain
                if bulk_phys_tag == upper_domain_tag:
                    # Check for CohesiveInterface connection
                    if cohesive_interface_tag is not None and cohesive_interface_tag in node_line_tags:
                        updated_node_ids[i] = interface_node_duplicates['CohesiveInterface'][original_node_id]
                    elif cohesive_particle_tag is not None and cohesive_particle_tag in node_line_tags:
                        updated_node_ids[i] = interface_node_duplicates['CohesiveParticle'][original_node_id]
                        
                # Case 2: Bulk element is LowerDomain
                elif bulk_phys_tag == lower_domain_tag:
                    if cohesive_interface_tag is not None and cohesive_interface_tag in node_line_tags:
                        # This node is on the main Y=0 interface and belongs to LowerDomain.
                        # It should connect to the original side of CohesiveInterface (which is its current original ID).
                        pass # No update needed
                    elif cohesive_particle_tag is not None and cohesive_particle_tag in node_line_tags:
                        # This node is on the particle interface and belongs to LowerDomain.
                        # It should connect to the duplicated side of CohesiveParticle.
                        updated_node_ids[i] = interface_node_duplicates['CohesiveParticle'][original_node_id]

                # Case 3: Bulk element is ParticleDomain
                elif bulk_phys_tag == particle_domain_tag:
                    if cohesive_particle_tag is not None and cohesive_particle_tag in node_line_tags:
                        # This node is on the particle interface and belongs to ParticleDomain.
                        # It should connect to the original side of CohesiveParticle (which is its current original ID).
                        pass # No update needed

            final_bulk_elements_connectivity.append(updated_node_ids)
            elements_final.append({
                'id': elem_data['id'],
                'type': elem_data['type'],
                'phys_tag': elem_data['phys_tag'],
                'node_ids': updated_node_ids
            })
        else:
            # For non-bulk elements, just copy them as is (their node IDs should be untouched or handled by cohesive logic)
            elements_final.append(elem_data)


    # Finalize dofs array
    dofs_array = np.zeros((max_node_id, 2), dtype=int)
    for node_id in range(1, max_node_id + 1):
        dofs_array[node_id - 1, 0] = (node_id - 1) * 2
        dofs_array[node_id - 1, 1] = (node_id - 1) * 2 + 1


    # Process other 1D physical lines for output
    final_other_line_nodes = {}
    for line_name, node_ids_list in line_nodes_by_physical_name_temp.items():
        if node_ids_list:
            unique_nodes = np.array(list(set(node_ids_list)), dtype=int) - 1 # 0-indexed and unique
            
            # For other lines, default to X-coordinate sorting
            x_coords = coor[unique_nodes, 0]
            sort_indices = np.argsort(x_coords)
            final_other_line_nodes[line_name] = unique_nodes[sort_indices]
        else:
            final_other_line_nodes[line_name] = np.array([], dtype=int)


    # Prepare final output dictionary
    results = {
        'coor': coor,
        'dofs': dofs_array,
        'elements': elements_final, # Contains all elements, with updated node IDs for bulk
        'physical_names': physical_names,
        'conn_bulk': np.array(final_bulk_elements_connectivity, dtype=int) - 1 if final_bulk_elements_connectivity else np.array([], dtype=int).reshape(0,4),
    }

    for cohesive_name in cohesive_line_names:
        conn_list = cohesive_elements_connectivity_list_by_name[cohesive_name]
        results[f'conn_{cohesive_name}'] = np.array(conn_list, dtype=int) - 1 if conn_list else np.array([], dtype=int).reshape(0,4)

    for line_name, nodes_array in final_other_line_nodes.items():
        results[line_name] = nodes_array

    return results