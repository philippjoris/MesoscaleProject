import numpy as np
import collections
import os
import math

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
            - 'ParticleLine': numpy.ndarray of 0-indexed original node IDs that form the CohesiveParticle interface.
            - Other physical line names as keys, with values being numpy.ndarray of 0-indexed node IDs,
              sorted by X-coordinate or traversed order as appropriate.
              These include all 1D physical lines EXCEPT the cohesive lines.
    """

    temp_nodes_dict = {}
    elements_raw = [] 
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
                if line.isdigit(): 
                    pass
                else:
                    parts = line.split()
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    temp_nodes_dict[node_id] = np.array([x, y, z])
                    if node_id > max_node_id: max_node_id = node_id
            elif current_section == '$Elements':
                if line.isdigit(): 
                    pass
                else:
                    parts = list(map(int, line.split()))
                    elem_id = parts[0]
                    elem_type = parts[1]
                    num_tags = parts[2]
                    phys_tag = parts[3] if num_tags >= 1 else 0
                    node_ids = parts[3 + num_tags:]

                    elements_raw.append({ 
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

    # Convert original nodes dictionary to numpy array (0-indexed) and substract 1 from node_id
    coor = np.zeros((max_node_id, 2), dtype=float)
    for node_id, coords_xyz in temp_nodes_dict.items():
        coor[node_id - 1, 0] = coords_xyz[0]
        coor[node_id - 1, 1] = coords_xyz[1]

    num_original_nodes = coor.shape[0]

    # This will store {node_id: set_of_physical_line_tags_it_belongs_to}
    # With this step we know to which physical domains each node belongs to.
    node_to_physical_line_tags = {}
    for elem_data in elements_raw:
        if elem_data['type'] == 1: # It's a 1D line element
            for node_id in elem_data['node_ids']:
                if node_id not in node_to_physical_line_tags:
                    node_to_physical_line_tags[node_id] = set()
                node_to_physical_line_tags[node_id].add(elem_data['phys_tag'])
    
    # Map: {original_gmsh_node_id: {interface_name: duplicated_node_id}}
    # Preparation for node duplication
    interface_node_duplicates = {name: {} for name in cohesive_line_names}
    interface_original_side_nodes = {name: {} for name in cohesive_line_names}

    next_available_node_id = max_node_id + 1
    new_coords_list = [] 

    # Tags for quick lookup
    cohesive_interface_tag = next((tag for tag, name in physical_names.items() if name == "CohesiveInterface"), None)
    cohesive_particle_tag = next((tag for tag, name in physical_names.items() if name == "CohesiveParticle"), None)

    # Pre-populate `interface_original_side_nodes` and `interface_node_duplicates`
    # This step correctly handles the "splitting" of original Gmsh nodes at junctions.
    for original_gmsh_node_id in range(1, num_original_nodes + 1):
        node_line_tags = node_to_physical_line_tags.get(original_gmsh_node_id, set())

        is_on_cohesive_interface = cohesive_interface_tag is not None and cohesive_interface_tag in node_line_tags
        is_on_cohesive_particle = cohesive_particle_tag is not None and cohesive_particle_tag in node_line_tags

        # Duplicating nodes on cohesive interface
        if is_on_cohesive_interface:
            # store the original node of the node on cohesiveInterface. This will be the bottom part of the interface later
            interface_original_side_nodes["CohesiveInterface"][original_gmsh_node_id] = original_gmsh_node_id
            
            # store a duplicated node of the node on cohesiveInterface. This will be the top part of the interface later
            if original_gmsh_node_id not in interface_node_duplicates["CohesiveInterface"]:
                interface_node_duplicates["CohesiveInterface"][original_gmsh_node_id] = next_available_node_id
                new_coords_list.append(coor[original_gmsh_node_id - 1])
                next_available_node_id += 1

        # Duplicating nodes on cohesive Particle. This will be the node belonging to the matrix bulk later.
        if is_on_cohesive_particle:
            interface_original_side_nodes["CohesiveParticle"][original_gmsh_node_id] = original_gmsh_node_id    
            
            # store a duplicated node of the node on cohesiveParticle. . This will be the node belonging to the particle bulk later.
            if original_gmsh_node_id not in interface_node_duplicates["CohesiveParticle"]:
                interface_node_duplicates["CohesiveParticle"][original_gmsh_node_id] = next_available_node_id
                new_coords_list.append(coor[original_gmsh_node_id - 1])
                next_available_node_id += 1

    # Notice: If a node is both on Cohesive Particle and Cohesive Interface, the node is dublicated twice!

    num_original_nodes_initial = coor.shape[0] 
    total_new_nodes = len(new_coords_list)
    
    final_coor = np.zeros((num_original_nodes_initial + total_new_nodes, 2), dtype=float)
    final_coor[:num_original_nodes_initial, :] = coor 

    for i, coords in enumerate(new_coords_list):
        final_coor[num_original_nodes_initial + i, :] = coords
    coor = final_coor 
    max_node_id = coor.shape[0] 

    # --- Construct Cohesive Elements Connectivity ---
    cohesive_elements_connectivity_list_by_name = {name: [] for name in cohesive_line_names}
    
    original_particle_line_nodes_set = set() 

    for interface_name in cohesive_line_names:
        current_cohesive_elements_conn = cohesive_elements_connectivity_list_by_name[interface_name]
        current_duplicated_node_map = interface_node_duplicates[interface_name]
        current_original_side_node_map = interface_original_side_nodes[interface_name]

        if interface_name == "CohesiveInterface":
            # Sort cohesive line elements by the x-coordinate of their first node for consistent local connectivity (left to right)
            cohesive_line_elements_by_name[interface_name].sort(key=lambda x: coor[x['nodes'][0] - 1, 0] if x['nodes'] else float('inf'))

            for line_elem in cohesive_line_elements_by_name[interface_name]:
                n1_gmsh_orig, n2_gmsh_orig = sorted(line_elem['nodes'], key=lambda n_id: coor[n_id - 1, 0])  

                # id_bl and id_br come from the 'original side' map
                id_bl = current_original_side_node_map.get(n1_gmsh_orig, n1_gmsh_orig) 
                id_br = current_original_side_node_map.get(n2_gmsh_orig, n2_gmsh_orig)

                # id_tl and id_tr come from the 'duplicated' map
                id_tl = current_duplicated_node_map[n1_gmsh_orig] 
                id_tr = current_duplicated_node_map[n2_gmsh_orig]

                current_cohesive_elements_conn.append([id_bl, id_br, id_tr, id_tl]) # Node IDs are 1-indexed at this stage

        elif interface_name == "CohesiveParticle":

            # 1. Get all unique nodes belonging to CohesiveParticle lines
            particle_line_node_ids = set()
            for line_elem in cohesive_line_elements_by_name[interface_name]:
                particle_line_node_ids.update(line_elem['nodes'])

            # 2. Calculate average center of the particle boundary nodes
            # This serves as the reference point for determining 'outward'
            particle_center_x = 0.0
            particle_center_y = 0.0
            for item in particle_line_node_ids:
                particle_center_x += coor[item - 1, 0]
                particle_center_y += coor[item - 1, 1]
            num_particle_nodes = len(particle_line_node_ids)
            
            if num_particle_nodes > 0:
                particle_center = np.array([particle_center_x / num_particle_nodes, particle_center_y / num_particle_nodes])
            else:
                # Fallback if no particle cohesive elements are found (should not happen if this block is entered)
                particle_center = np.array([0.0, 0.0])
                print("Warning: No nodes found for CohesiveParticle interface to calculate center.")


            # 3. Build the connected loop(s) of elements 
            particle_node_to_elements = collections.defaultdict(list)
            for line_elem in cohesive_line_elements_by_name[interface_name]:
                for node_id in line_elem['nodes']:
                    particle_node_to_elements[node_id].append(line_elem)

            all_particle_elements = list(cohesive_line_elements_by_name[interface_name])
            processed_element_ids = set()
            
            all_ordered_particle_contours = []

            while len(processed_element_ids) < len(all_particle_elements):
                start_elem = None
                for elem in all_particle_elements:
                    if elem['id'] not in processed_element_ids:
                        start_elem = elem
                        break
                
                if start_elem is None:
                    break 

                start_node_for_contour = None
                # For a closed loop, all nodes have degree 2. Pick the 'lowest-leftmost' to start a consistent traversal.
                if len(start_elem['nodes']) == 2: 
                    potential_start_nodes = [n for n in start_elem['nodes'] if len(particle_node_to_elements[n]) == 1]
                    if potential_start_nodes: 
                        start_node_for_contour = potential_start_nodes[0]
                    else: 
                        start_node_for_contour = sorted(start_elem['nodes'], key=lambda n_id: (coor[n_id - 1, 0], coor[n_id - 1, 1]))[0]
                else:
                    # Handle cases where start_elem is not a 2-node line, or has unusual structure
                    print(f"Warning: start_elem {start_elem['id']} has unexpected number of nodes or structure.")
                    break 


                current_node_id = start_node_for_contour
                # Determine the initial direction from start_node_for_contour to the other node in start_elem
                previous_node_id = next(n for n in start_elem['nodes'] if n != start_node_for_contour)

                current_contour_ordered_elements = []

                # Traversal loop
                while True:
                    # Find elements connected to current_node_id that are not the previous_node_id and not processed
                    candidate_elements = [e for e in particle_node_to_elements[current_node_id] 
                                          if e['id'] not in processed_element_ids]

                    # Filter out the element that were already checked
                    next_elem = None
                    for cand_elem in candidate_elements:
                        if (current_node_id == cand_elem['nodes'][0] and previous_node_id == cand_elem['nodes'][1]) or \
                           (current_node_id == cand_elem['nodes'][1] and previous_node_id == cand_elem['nodes'][0]):
                            continue
                        
                        # Find the other node in this candidate element
                        other_node_in_cand = next(n for n in cand_elem['nodes'] if n != current_node_id)
                        
                        if other_node_in_cand != previous_node_id:
                            next_elem = cand_elem
                            break
                    
                    if next_elem is None: # No more elements to follow in this contour
                        break 

                    current_contour_ordered_elements.append(next_elem)
                    processed_element_ids.add(next_elem['id'])

                    old_current_node_id = current_node_id
                    current_node_id = next(n for n in next_elem['nodes'] if n != current_node_id)
                    previous_node_id = old_current_node_id 

                if current_contour_ordered_elements:
                    all_ordered_particle_contours.append(current_contour_ordered_elements)

            # Process each ordered contour
            for contour_elements in all_ordered_particle_contours:
                for line_elem in contour_elements:
                    n1_orig, n2_orig = line_elem['nodes']

                    # Calculate midpoint of this line element (or just its average node coord)
                    mid_x = (coor[n1_orig - 1, 0] + coor[n2_orig - 1, 0]) / 2.0
                    mid_y = (coor[n1_orig - 1, 1] + coor[n2_orig - 1, 1]) / 2.0
                    elem_midpoint = np.array([mid_x, mid_y])

                    # Vector from particle center to element midpoint
                    V_radial = elem_midpoint - particle_center
                    V_radial_norm = np.linalg.norm(V_radial)
                    if V_radial_norm > 1e-12:
                        V_radial_unit = V_radial / V_radial_norm
                    else:
                        V_radial_unit = np.array([0.0, 1.0]) # Fallback: default to upward (should be rare)

                    # Determine initial tangent assuming (n1_orig, n2_orig) order
                    t_initial_x = coor[n2_orig - 1, 0] - coor[n1_orig - 1, 0]
                    t_initial_y = coor[n2_orig - 1, 1] - coor[n1_orig - 1, 1]

                    # Calculate the normal that your C++ code would produce from this tangent: n = [-t_y, t_x]
                    n_calc_x = -t_initial_y
                    n_calc_y = t_initial_x

                    # Check if this calculated normal points in the desired outward direction
                    dot_product = n_calc_x * V_radial_unit[0] + n_calc_y * V_radial_unit[1]

                    actual_n1_gmsh_orig, actual_n2_gmsh_orig = n1_orig, n2_orig # Keep original order    

                    # Check if node is at cross section of cohesiveInterface and cohesiveParticle and check if the 
                    # element in question is above the cohesive interface
                    n1check = interface_node_duplicates["CohesiveInterface"].get(actual_n1_gmsh_orig)
                    n2check = interface_node_duplicates["CohesiveInterface"].get(actual_n2_gmsh_orig)

                    if n2check is not None and elem_midpoint[1] > 0:
                        id_tl = interface_node_duplicates["CohesiveInterface"][actual_n2_gmsh_orig]
                    else:
                        id_tl = current_original_side_node_map.get(actual_n2_gmsh_orig, actual_n2_gmsh_orig)

                    if n1check is not None and elem_midpoint[1] > 0:
                        id_tr = interface_node_duplicates["CohesiveInterface"][actual_n1_gmsh_orig]
                    else:    
                        id_tr = current_original_side_node_map.get(actual_n1_gmsh_orig, actual_n1_gmsh_orig) 
                    

                    id_br = current_duplicated_node_map[actual_n1_gmsh_orig] 
                    id_bl = current_duplicated_node_map[actual_n2_gmsh_orig]

                    # Append the 1-indexed cohesive element connectivity
                    current_cohesive_elements_conn.append([id_bl, id_br, id_tr, id_tl])
                    original_particle_line_nodes_set.add(id_bl)
                    original_particle_line_nodes_set.add(id_br)


    # --- Update Bulk Elements Connectivity ---    
    matrix_elements_connectivity = []
    particle_elements_connectivity = []
    elements_final = [] 

    upper_domain_tag = next( (tag for tag, name in physical_names.items() if name == "UpperDomain"), None)
    lower_domain_tag = next( (tag for tag, name in physical_names.items() if name == "LowerDomain"), None)
    particle_domain_tag = next( (tag for tag, name in physical_names.items() if name == "ParticleDomain"), None)


    for elem_data in elements_raw: # Iterate through original parsed elements
        if elem_data['type'] == 3: # This is a 4-node quad (bulk element)
            original_gmsh_node_ids = elem_data['node_ids']
            updated_node_ids = list(original_gmsh_node_ids) 
            bulk_phys_tag = elem_data['phys_tag']

            for i, original_gmsh_node_id in enumerate(original_gmsh_node_ids):
                if original_gmsh_node_id == 8:
                    a=1
                node_line_tags = node_to_physical_line_tags.get(original_gmsh_node_id, set())

                mapped_node_id = original_gmsh_node_id # Default

                # Case 1: Bulk element is UpperDomain
                if bulk_phys_tag == upper_domain_tag:
                    # UpperDomain connects to the *duplicated* side of any interface it touches
                    if cohesive_particle_tag is not None and cohesive_interface_tag is not None and \
                        cohesive_particle_tag in node_line_tags and cohesive_interface_tag in node_line_tags:
                        if original_gmsh_node_id in interface_node_duplicates['CohesiveInterface']:
                            mapped_node_id = interface_node_duplicates['CohesiveInterface'][original_gmsh_node_id]   
                    elif cohesive_interface_tag is not None and cohesive_interface_tag in node_line_tags:
                        if original_gmsh_node_id in interface_node_duplicates['CohesiveInterface']:
                            mapped_node_id = interface_node_duplicates['CohesiveInterface'][original_gmsh_node_id] 
                    elif cohesive_particle_tag is not None and cohesive_particle_tag in node_line_tags:
                        if original_gmsh_node_id in interface_original_side_nodes['CohesiveParticle']:
                            mapped_node_id = interface_original_side_nodes['CohesiveParticle'][original_gmsh_node_id] 

                # Case 2: Bulk element is LowerDomain
                elif bulk_phys_tag == lower_domain_tag:
                    # LowerDomain connects to the *original* side of any interface it touches
                    if cohesive_particle_tag is not None and cohesive_interface_tag is not None and \
                        cohesive_particle_tag in node_line_tags and cohesive_interface_tag in node_line_tags:
                        if original_gmsh_node_id in interface_original_side_nodes['CohesiveInterface']:
                            mapped_node_id = interface_original_side_nodes['CohesiveInterface'][original_gmsh_node_id]                    
                    elif cohesive_particle_tag is not None and cohesive_particle_tag in node_line_tags:
                        if original_gmsh_node_id in interface_original_side_nodes['CohesiveParticle']:
                            mapped_node_id = interface_original_side_nodes['CohesiveParticle'][original_gmsh_node_id]
                    elif cohesive_interface_tag is not None and cohesive_interface_tag in node_line_tags:
                        if original_gmsh_node_id in interface_original_side_nodes['CohesiveInterface']:
                            mapped_node_id = interface_original_side_nodes['CohesiveInterface'][original_gmsh_node_id]
                
                # Case 3: Bulk element is ParticleDomain
                elif bulk_phys_tag == particle_domain_tag:
                    # ParticleDomain always connects to the duplicated (particle) side of CohesiveParticle 
                    if cohesive_particle_tag is not None and cohesive_particle_tag in node_line_tags:
                        if original_gmsh_node_id in interface_node_duplicates['CohesiveParticle']:
                            mapped_node_id = interface_node_duplicates['CohesiveParticle'][original_gmsh_node_id]
                
                updated_node_ids[i] = mapped_node_id

            if bulk_phys_tag == particle_domain_tag:
                particle_elements_connectivity.append(updated_node_ids)
            else:
                matrix_elements_connectivity.append(updated_node_ids)
            elements_final.append({
                'id': elem_data['id'],
                'type': elem_data['type'],
                'phys_tag': elem_data['phys_tag'],
                'node_ids': updated_node_ids
            })
        else:
            pass


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
        'physical_names': physical_names,
        'conn_matrix': np.array(matrix_elements_connectivity, dtype=int) - 1 if matrix_elements_connectivity else np.array([], dtype=int).reshape(0,4),
        'conn_particle': np.array(particle_elements_connectivity, dtype=int) - 1 if particle_elements_connectivity else np.array([], dtype=int).reshape(0,4),
    }

    for cohesive_name in cohesive_line_names:
        conn_list = cohesive_elements_connectivity_list_by_name[cohesive_name]
        results[f'conn_{cohesive_name}'] = np.array(conn_list, dtype=int) - 1 if conn_list else np.array([], dtype=int).reshape(0,4)

    for line_name, nodes_array in final_other_line_nodes.items():
        results[line_name] = nodes_array

    # Add the derived ParticleLine nodes (0-indexed)
    if original_particle_line_nodes_set:
        particle_line_nodes_array = np.array(list(original_particle_line_nodes_set), dtype=int) - 1 # Convert to 0-indexed
        
        # Sort these nodes by X-coordinate for consistency
        if particle_line_nodes_array.size > 0:
            x_coords_particle_line = coor[particle_line_nodes_array, 0]
            sort_indices_particle_line = np.argsort(x_coords_particle_line)
            results["ParticleLine"] = particle_line_nodes_array[sort_indices_particle_line]
        else:
            results["ParticleLine"] = np.array([], dtype=int)
    else:
        results["ParticleLine"] = np.array([], dtype=int)

    return results

def main():
    mesh_file_name = "i.geo_simple_bottom.msh2"
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_file_path = os.path.join(curr_dir, mesh_file_name)

    if os.path.exists(mesh_file_path):
        print(f"Parsing mesh: {mesh_file_path}")
        mesh = parse_msh( 
            msh_filepath=mesh_file_path
        )

main()