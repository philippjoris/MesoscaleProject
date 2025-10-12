import numpy as np
import collections
import os

def parse_msh(msh_filepath, cohesive_surface_names=["CohesiveInterface", "CohesiveParticle"], tolerance=1e-3):
    """
    Parses a Gmsh MSH file (version 2.2 ASCII) for a 3D mesh with 8-noded hexahedral bulk elements
    and 4-noded quadrilateral cohesive surface elements. Duplicates nodes on cohesive surfaces,
    constructs 8-noded cohesive hexahedra, and groups surface nodes into unique sets, ensuring
    no node appears in more than one set.

    Args:
        msh_filepath (str): Path to the Gmsh MSH file.
        cohesive_surface_names (list): Names of physical surfaces representing cohesive interfaces.
        tolerance (float): Tolerance for matching node coordinates.

    Returns:
        dict: Containing:
            - 'coor': numpy.ndarray (num_nodes, 3) with [x, y, z] coordinates (0-indexed node IDs).
            - 'dofs': numpy.ndarray (num_nodes, 3) with [dof_x, dof_y, dof_z] for each node.
            - 'elements': List of dicts [{id, type, phys_tag, node_ids}] with updated node IDs.
            - 'physical_names': Dict {tag_id: name}.
            - 'conn_CohesiveInterface': numpy.ndarray (N, 8) for cohesive hexahedra (0-indexed).
            - 'conn_CohesiveParticle': numpy.ndarray (M, 8) for cohesive hexahedra (0-indexed).
            - 'conn_bulk': numpy.ndarray (P, 8) for bulk hexahedra (0-indexed).
            - 'CohesiveInterface': numpy.ndarray of node IDs on CohesiveInterface (original nodes).
            - 'ParticleSurface': numpy.ndarray of node IDs on CohesiveParticle (duplicated nodes).
            - Other physical surface names: numpy.ndarray of node IDs, all sets disjoint.
    """
    temp_nodes_dict = {}
    elements_raw = []
    physical_names = {}
    cohesive_surface_elements_by_name = {name: [] for name in cohesive_surface_names}
    surface_nodes_by_physical_name_temp = {}
    max_node_id = 0
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
                    if name in cohesive_surface_names:
                        cohesive_tags.add(tag_id)
            elif current_section == '$Nodes':
                if not line.isdigit():
                    parts = line.split()
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    temp_nodes_dict[node_id] = np.array([x, y, z])
                    if node_id > max_node_id:
                        max_node_id = node_id
            elif current_section == '$Elements':
                if not line.isdigit():
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

                    if elem_type == 3:  # 4-noded quad (surface element)
                        surface_name = physical_names.get(phys_tag)
                        if surface_name in cohesive_surface_names:
                            cohesive_surface_elements_by_name[surface_name].append({
                                'id': elem_id,
                                'nodes': node_ids,
                                'phys_tag': phys_tag
                            })
                        elif surface_name:
                            if surface_name not in surface_nodes_by_physical_name_temp:
                                surface_nodes_by_physical_name_temp[surface_name] = []
                            surface_nodes_by_physical_name_temp[surface_name].extend(node_ids)

    # Convert nodes to numpy array (0-indexed)
    coor = np.zeros((max_node_id, 3), dtype=float)
    for node_id, coords in temp_nodes_dict.items():
        coor[node_id - 1] = coords
    num_original_nodes = coor.shape[0]

    # Map nodes to physical surface tags
    node_to_physical_surface_tags = {}
    for elem_data in elements_raw:
        if elem_data['type'] == 3:
            for node_id in elem_data['node_ids']:
                if node_id not in node_to_physical_surface_tags:
                    node_to_physical_surface_tags[node_id] = set()
                node_to_physical_surface_tags[node_id].add(elem_data['phys_tag'])

    # Node duplication for cohesive surfaces
    interface_node_duplicates = {name: {} for name in cohesive_surface_names}
    interface_original_side_nodes = {name: {} for name in cohesive_surface_names}
    next_available_node_id = max_node_id + 1
    new_coords_list = []

    cohesive_interface_tag = next((tag for tag, name in physical_names.items() if name == "CohesiveInterface"), None)
    cohesive_particle_tag = next((tag for tag, name in physical_names.items() if name == "CohesiveParticle"), None)

    for original_gmsh_node_id in range(1, num_original_nodes + 1):
        node_surface_tags = node_to_physical_surface_tags.get(original_gmsh_node_id, set())
        is_on_cohesive_interface = cohesive_interface_tag in node_surface_tags
        is_on_cohesive_particle = cohesive_particle_tag in node_surface_tags

        if is_on_cohesive_interface:
            interface_original_side_nodes["CohesiveInterface"][original_gmsh_node_id] = original_gmsh_node_id
            if original_gmsh_node_id not in interface_node_duplicates["CohesiveInterface"]:
                interface_node_duplicates["CohesiveInterface"][original_gmsh_node_id] = next_available_node_id
                new_coords_list.append(coor[original_gmsh_node_id - 1])
                next_available_node_id += 1

        if is_on_cohesive_particle:
            interface_original_side_nodes["CohesiveParticle"][original_gmsh_node_id] = original_gmsh_node_id
            if original_gmsh_node_id not in interface_node_duplicates["CohesiveParticle"]:
                interface_node_duplicates["CohesiveParticle"][original_gmsh_node_id] = next_available_node_id
                new_coords_list.append(coor[original_gmsh_node_id - 1])
                next_available_node_id += 1

    # Update coordinates array
    total_new_nodes = len(new_coords_list)
    final_coor = np.zeros((num_original_nodes + total_new_nodes, 3), dtype=float)
    final_coor[:num_original_nodes] = coor
    for i, coords in enumerate(new_coords_list):
        final_coor[num_original_nodes + i] = coords
    coor = final_coor
    max_node_id = coor.shape[0]

    # Construct cohesive element connectivity
    cohesive_elements_connectivity_list_by_name = {name: [] for name in cohesive_surface_names}
    original_interface_nodes_set = set()  # For CohesiveInterface nodes
    original_particle_surface_nodes_set = set()

    def get_centroid(elem):
        nodes = elem['nodes']
        centroid = np.mean([coor[node_id - 1] for node_id in nodes], axis=0)
        return centroid[0]

    for interface_name in cohesive_surface_names:
        current_cohesive_elements_conn = cohesive_elements_connectivity_list_by_name[interface_name]
        current_duplicated_node_map = interface_node_duplicates[interface_name]
        current_original_side_node_map = interface_original_side_nodes[interface_name]

        if interface_name == "CohesiveInterface":
            cohesive_surface_elements_by_name[interface_name].sort(key=get_centroid)
            for quad_elem in cohesive_surface_elements_by_name[interface_name]:
                n1, n2, n3, n4 = quad_elem['nodes']
                id_b1 = current_original_side_node_map.get(n1, n1)
                id_b2 = current_original_side_node_map.get(n2, n2)
                id_b3 = current_original_side_node_map.get(n3, n3)
                id_b4 = current_original_side_node_map.get(n4, n4)
                id_t1 = current_duplicated_node_map[n1]
                id_t2 = current_duplicated_node_map[n2]
                id_t3 = current_duplicated_node_map[n3]
                id_t4 = current_duplicated_node_map[n4]
                current_cohesive_elements_conn.append([id_b1, id_b2, id_b3, id_b4, id_t1, id_t2, id_t3, id_t4])
                original_interface_nodes_set.update([id_b1, id_b2, id_b3, id_b4])

        elif interface_name == "CohesiveParticle":
            particle_node_ids = set()
            for quad_elem in cohesive_surface_elements_by_name[interface_name]:
                particle_node_ids.update(quad_elem['nodes'])
            particle_center = np.mean([coor[node_id - 1] for node_id in particle_node_ids], axis=0) if particle_node_ids else np.array([0.0, 0.0, 0.0])

            quad_to_quads = collections.defaultdict(list)
            for quad_elem in cohesive_surface_elements_by_name[interface_name]:
                nodes = set(quad_elem['nodes'])
                for other_elem in cohesive_surface_elements_by_name[interface_name]:
                    if other_elem['id'] != quad_elem['id'] and len(nodes.intersection(other_elem['nodes'])) >= 2:
                        quad_to_quads[quad_elem['id']].append(other_elem)

            all_ordered_quads = []
            processed_ids = set()
            while len(processed_ids) < len(cohesive_surface_elements_by_name[interface_name]):
                start_elem = next((e for e in cohesive_surface_elements_by_name[interface_name] if e['id'] not in processed_ids), None)
                if not start_elem:
                    break
                current_ordered_quads = [start_elem]
                processed_ids.add(start_elem['id'])
                while True:
                    current_id = current_ordered_quads[-1]['id']
                    neighbors = [e for e in quad_to_quads[current_id] if e['id'] not in processed_ids]
                    if not neighbors:
                        break
                    next_elem = neighbors[0]
                    current_ordered_quads.append(next_elem)
                    processed_ids.add(next_elem['id'])
                all_ordered_quads.append(current_ordered_quads)

            for quad_patch in all_ordered_quads:
                for quad_elem in quad_patch:
                    n1, n2, n3, n4 = quad_elem['nodes']
                    v1 = coor[n2 - 1] - coor[n1 - 1]
                    v2 = coor[n4 - 1] - coor[n1 - 1]
                    normal = np.cross(v1, v2)
                    centroid = np.mean([coor[n - 1] for n in [n1, n2, n3, n4]], axis=0)
                    radial = centroid - particle_center
                    if np.dot(normal, radial) < 0:
                        n1, n2, n3, n4 = n1, n4, n3, n2
                    id_b1 = current_duplicated_node_map[n1]
                    id_b2 = current_duplicated_node_map[n2]
                    id_b3 = current_duplicated_node_map[n3]
                    id_b4 = current_duplicated_node_map[n4]
                    id_t1 = current_original_side_node_map.get(n1, n1)
                    id_t2 = current_original_side_node_map.get(n2, n2)
                    id_t3 = current_original_side_node_map.get(n3, n3)
                    id_t4 = current_original_side_node_map.get(n4, n4)
                    current_cohesive_elements_conn.append([id_b1, id_b2, id_b3, id_b4, id_t1, id_t2, id_t3, id_t4])
                    original_particle_surface_nodes_set.update([id_b1, id_b2, id_b3, id_b4])

    # Update bulk element connectivity
    final_bulk_elements_connectivity = []
    elements_final = []
    upper_domain_tag = next((tag for tag, name in physical_names.items() if name == "UpperDomain"), None)
    lower_domain_tag = next((tag for tag, name in physical_names.items() if name == "LowerDomain"), None)
    particle_domain_tag = next((tag for tag, name in physical_names.items() if name == "ParticleDomain"), None)

    for elem_data in elements_raw:
        if elem_data['type'] == 5:
            original_gmsh_node_ids = elem_data['node_ids']
            updated_node_ids = list(original_gmsh_node_ids)
            bulk_phys_tag = elem_data['phys_tag']
            for i, original_gmsh_node_id in enumerate(original_gmsh_node_ids):
                node_surface_tags = node_to_physical_surface_tags.get(original_gmsh_node_id, set())
                mapped_node_id = original_gmsh_node_id
                if bulk_phys_tag == upper_domain_tag:
                    if cohesive_interface_tag in node_surface_tags and original_gmsh_node_id in interface_node_duplicates['CohesiveInterface']:
                        mapped_node_id = interface_node_duplicates['CohesiveInterface'][original_gmsh_node_id]
                    elif cohesive_particle_tag in node_surface_tags and original_gmsh_node_id in interface_original_side_nodes['CohesiveParticle']:
                        mapped_node_id = interface_original_side_nodes['CohesiveParticle'][original_gmsh_node_id]
                elif bulk_phys_tag == lower_domain_tag:
                    if cohesive_interface_tag in node_surface_tags and original_gmsh_node_id in interface_original_side_nodes['CohesiveInterface']:
                        mapped_node_id = interface_original_side_nodes['CohesiveInterface'][original_gmsh_node_id]
                    elif cohesive_particle_tag in node_surface_tags and original_gmsh_node_id in interface_original_side_nodes['CohesiveParticle']:
                        mapped_node_id = interface_original_side_nodes['CohesiveParticle'][original_gmsh_node_id]
                elif bulk_phys_tag == particle_domain_tag:
                    if cohesive_particle_tag in node_surface_tags and original_gmsh_node_id in interface_node_duplicates['CohesiveParticle']:
                        mapped_node_id = interface_node_duplicates['CohesiveParticle'][original_gmsh_node_id]
                updated_node_ids[i] = mapped_node_id   
            updated_node_ids = [updated_node_ids[i] for i in [2,6,7,3,1,5,4,0]]
            final_bulk_elements_connectivity.append(updated_node_ids)
            elements_final.append({
                'id': elem_data['id'],
                'type': elem_data['type'],
                'phys_tag': elem_data['phys_tag'],
                'node_ids': updated_node_ids
            })

    # DOFs array
    dofs_array = np.zeros((max_node_id, 3), dtype=int)
    for node_id in range(1, max_node_id + 1):
        dofs_array[node_id - 1] = [(node_id - 1) * 3, (node_id - 1) * 3 + 1, (node_id - 1) * 3 + 2]

    # Process node sets and ensure uniqueness
    final_node_sets = {}

    # CohesiveInterface nodes (original nodes, bottom face)
    if cohesive_elements_connectivity_list_by_name["CohesiveInterface"]:
        interface_nodes = np.array(list(original_interface_nodes_set), dtype=int) - 1
        if interface_nodes.size > 0:
            x_coords = coor[interface_nodes, 0]
            sort_indices = np.argsort(x_coords)
            final_node_sets["CohesiveInterface"] = interface_nodes[sort_indices]
        else:
            final_node_sets["CohesiveInterface"] = np.array([], dtype=int)
    else:
        final_node_sets["CohesiveInterface"] = np.array([], dtype=int)

    # ParticleSurface nodes (duplicated nodes, particle side)
    if original_particle_surface_nodes_set:
        particle_nodes = np.array(list(original_particle_surface_nodes_set), dtype=int) - 1
        if particle_nodes.size > 0:
            x_coords = coor[particle_nodes, 0]
            sort_indices = np.argsort(x_coords)
            final_node_sets["ParticleSurface"] = particle_nodes[sort_indices]
        else:
            final_node_sets["ParticleSurface"] = np.array([], dtype=int)
    else:
        final_node_sets["ParticleSurface"] = np.array([], dtype=int)

    # Other physical surfaces
    for surface_name, node_ids_list in surface_nodes_by_physical_name_temp.items():
        if node_ids_list:
            unique_nodes = np.array(list(set(node_ids_list)), dtype=int) - 1
            x_coords = coor[unique_nodes, 0]
            sort_indices = np.argsort(x_coords)
            final_node_sets[surface_name] = unique_nodes[sort_indices]
        else:
            final_node_sets[surface_name] = np.array([], dtype=int)

    # Ensure node sets are disjoint
    node_set_priority = ["CohesiveInterface", "ParticleSurface"] + [s for s in final_node_sets if s not in ["CohesiveInterface", "ParticleSurface"]]
    for i, set_name in enumerate(node_set_priority[1:], 1):
        current_nodes = set(final_node_sets[set_name])
        # Remove nodes that appear in higher-priority sets
        for higher_priority_set in node_set_priority[:i]:
            current_nodes -= set(final_node_sets[higher_priority_set])
        # Update the node set, preserving order
        if current_nodes:
            current_nodes_array = np.array(list(current_nodes), dtype=int)
            x_coords = coor[current_nodes_array, 0]
            sort_indices = np.argsort(x_coords)
            final_node_sets[set_name] = current_nodes_array[sort_indices]
        else:
            final_node_sets[set_name] = np.array([], dtype=int)

    # Prepare output
    results = {
        'coor': coor,
        'dofs': dofs_array,
        'physical_names': physical_names,
        'conn_bulk': np.array(final_bulk_elements_connectivity, dtype=int) - 1 if final_bulk_elements_connectivity else np.array([], dtype=int).reshape(0, 8),
    }

    for cohesive_name in cohesive_surface_names:
        conn_list = cohesive_elements_connectivity_list_by_name[cohesive_name]
        results[f'conn_{cohesive_name}'] = np.array(conn_list, dtype=int) - 1 if conn_list else np.array([], dtype=int).reshape(0, 8)

    results.update(final_node_sets)

    return results

def main():
    mesh_file_name = "i.geo_3D_simple.msh2"
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_file_path = os.path.join(curr_dir, mesh_file_name)

    if os.path.exists(mesh_file_path):
        print(f"Parsing mesh: {mesh_file_path}")
        mesh = parse_msh(msh_filepath=mesh_file_path)
        print("Node coordinates shape:", mesh['coor'].shape)
        print("Bulk elements:", mesh['conn_bulk'].shape)
        for name in ["CohesiveInterface", "CohesiveParticle"]:
            print(f"{name} elements:", mesh[f'conn_{name}'].shape)
        for set_name in mesh:
            if isinstance(mesh[set_name], np.ndarray) and set_name not in ['coor', 'dofs', 'conn_bulk', 'conn_CohesiveInterface', 'conn_CohesiveParticle']:
                print(f"{set_name} nodes:", len(mesh[set_name]))

    a=1

if __name__ == "__main__":
    main()