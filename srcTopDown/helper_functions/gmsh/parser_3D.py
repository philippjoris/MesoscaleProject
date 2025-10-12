import numpy as np
import collections
import os
import json

def parse_msh(msh_filepath, json_filepath):
    """
    Parses a Gmsh MSH file (version 2.2 ASCII) to extract nodes and elements,
    identifies nodes belonging to specified cohesive physical lines,
    duplicates these nodes based on the *specific interface* they belong to,
    and constructs 4-node cohesive elements and updates bulk element connectivity.

    Args:
        msh_filepath (str): Path to the Gmsh MSH file.
        json_filepath (str): Path to the JSON support file.

    Returns:
        dict: A dictionary containing:
            - 'coor': numpy.ndarray of shape (num_nodes, 2) storing [x, y] coordinates for each node,
                      where row index (node_id - 1) corresponds to the node_id
            - 'dofs': numpy.ndarray of shape (num_nodes, 2) storing [dof_id_x, dof_id_y] for each node
            - 'elements': [{id: int, type: int, phys_tag: int, node_ids: list}] (all elements, with updated node IDs)
            - 'physical_names': {tag_id: name}
            - 'conn_particle': numpy.ndarray of shape (P, 8), where P is the number of particle bulk elements
                                        (node IDs are 0-indexed)
            - 'conn_interface': numpy.ndarray of shape (I, 8), where I is the number of interface bulk elements
                                        (node IDs are 0-indexed)
            - 'conn_matrix': numpy.ndarray of shape (M, 4), where M is the number of matrix bulk elements
                                        (node IDs are 0-indexed)
            - Other physical line names as keys, with values being numpy.ndarray of 0-indexed node IDs,
              sorted by X-coordinate or traversed order as appropriate.
    """

    temp_nodes_dict = {}
    elements_raw = [] 
    physical_names = {}
    surface_nodes_by_physical_name = {}
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
                    phys_tag = parts[4] if num_tags >= 1 else 0
                    node_ids = parts[3 + num_tags:]

                    elements_raw.append({ 
                        'id': elem_id,
                        'type': elem_type,
                        'phys_tag': phys_tag,
                        'node_ids': node_ids
                    })

    
    try:
        with open(json_filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_filepath}")
        return {}, {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return {}, {}

    elem_sets = data.get("element_sets", {})
    node_sets = data.get("node_sets", {})

    # --- 2. Process element sets (unchanged logic) ---
    elem_id_to_group = {}
    for group_name, elem_list in elem_sets.items():
        # group_name will be "matrix", "interface", "particle"
        for eid in elem_list:
            elem_id_to_group[eid] = group_name

    node_id_to_group = {}       
    for group_name, node_data in node_sets.items():
        if isinstance(node_data, list):
            # Handles lists of nodes (Faces, Edges)
            for nid in node_data:
                node_id_to_group[nid] = group_name
        elif isinstance(node_data, int):
            # Handles single corner node IDs (Integers)
            node_id_to_group[node_data] = group_name
        else:
            print(f"Warning: Node set '{group_name}' has unexpected type: {type(node_data)}")         

    # Convert original nodes dictionary to numpy array (0-indexed) and substract 1 from node_id
    coor = np.zeros((max_node_id, 3), dtype=float)
    for node_id, coords_xyz in temp_nodes_dict.items():
        coor[node_id - 1, 0] = coords_xyz[0]
        coor[node_id - 1, 1] = coords_xyz[1]
        coor[node_id - 1, 2] = coords_xyz[2]

    num_original_nodes = coor.shape[0]

    # --- Update Bulk Elements Connectivity ---    
    matrix_elem_conn = []
    interface_elem_conn = []
    particle_elem_conn = []
    elements_final = [] 

    # --- 1. Create Phyiscal Domains by JSON support file ---

    for elem_data in elements_raw:
        if elem_data['type'] == 5: # This is a 8-node hex (bulk element)
            elem_id = elem_data['id']
            node_ids  = elem_data['node_ids']
            updated_node_ids = list(node_ids) 
            phys_tag = elem_id_to_group.get(elem_id, None)

            # Assign connectivity based on physical group
            if phys_tag == "matrix":
                matrix_elem_conn.append(node_ids)
            elif phys_tag == "interface":
                interface_elem_conn.append(node_ids)
            elif phys_tag == "particle":
                particle_elem_conn.append(node_ids)
            else:
                pass  # or handle unassigned elements

            elements_final.append({
                'id': elem_data['id'],
                'type': elem_data['type'],
                'phys_tag': phys_tag,
                'node_ids': updated_node_ids
            })
        else:
            pass


    # Finalize dofs array
    dofs_array = np.zeros((max_node_id, 3), dtype=int)
    for node_id in range(1, max_node_id + 1):
        dofs_array[node_id - 1, 0] = (node_id - 1) * 3
        dofs_array[node_id - 1, 1] = (node_id - 1) * 3 + 1
        dofs_array[node_id - 1, 2] = (node_id - 1) * 3 + 2

    # --- NODE SETS ---
    final_node_sets = {}
    for surface_name, node_data in node_sets.items():
        if isinstance(node_data, list):
            if node_data:
                unique_nodes = np.array(list(set(node_data)), dtype=int) - 1
                x_coords = coor[unique_nodes, 0]
                sort_indices = np.argsort(x_coords)
                final_node_sets[surface_name] = unique_nodes[sort_indices]
            else:
                final_node_sets[surface_name] = np.array([], dtype=int)
        elif isinstance(node_data, int):
            unique_nodes = node_data -1
            final_node_sets[surface_name] = np.array([unique_nodes], dtype=int)
        else:
            final_node_sets[surface_name] = np.array([], dtype=int)            

    # Prepare final output dictionary
    results = {
        'coor': coor,
        'dofs': dofs_array,
        'physical_names': physical_names,
        'conn_matrix': np.array(matrix_elem_conn, dtype=int) - 1 if matrix_elem_conn else np.array([], dtype=int).reshape(0,8),
        'conn_interface': np.array(interface_elem_conn, dtype=int) - 1 if interface_elem_conn else np.array([], dtype=int).reshape(0,8),
        'conn_particle': np.array(particle_elem_conn, dtype=int) - 1 if particle_elem_conn else np.array([], dtype=int).reshape(0,8),
    }

    results.update(final_node_sets)

    return results

def main():
    mesh_file_name = "i.geo_composite_cube.msh2"
    json_file_name = "i.geo_element_sets.json"
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_file_path = os.path.join(curr_dir, mesh_file_name)
    json_file_path = os.path.join(curr_dir, json_file_name)

    if os.path.exists(mesh_file_path) and os.path.exists(json_file_path):
        print(f"Parsing mesh: {mesh_file_path}")
        mesh = parse_msh( 
            msh_filepath=mesh_file_path,
            json_filepath=json_file_path
        )

main()