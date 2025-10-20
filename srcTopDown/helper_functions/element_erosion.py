"""
file: element_erosion.py
author: Philipp van der Loos

A helper function for element erosion.

"""

import numpy as np
import copy

def element_erosionGooseMesh(Solver, vector, conn, coor, material, elem, elem0, fint_e, fext_init, disp, newly_failed, K, fe, I2, meshRefined):

    du = np.zeros_like(disp)
        
    # Create new external force vector per element
    new_fext_elem = np.zeros_like(fe)

    # Iterate over each newly failed element
    for failed in newly_failed:
        print(f"this are the newly failed elements {newly_failed}")
        for elem in meshRefined.map[failed]:
            mat.delete_element(failed)

        # -----------------------
        #
        #   multiple elements deleted at onces, what happens to the forces ?
        #   The following algorithm is only for single element deletions.
        #
        # -----------------------


        # Adaptive force application strategy
        force_multipliers_increase = [0.0, 0.25, 0.5, 0.75, 1.0] # More granular steps for increase
        
        converged_at_multiplier = None
        
        # Phase 1: Gradually apply internal forces to external forces
        for incr_factor in force_multipliers_increase:

            print(f"INFO: Trying to smooth forces with multiplier: {incr_factor:.2f} for element {failed}")
            
            disp_curr = disp.copy() 
            fext = fext_init.copy()
            new_fext_elem = np.zeros_like(fe)
            elem.update_x(vector.AsElement(coor + disp, conn))

            # Apply the current force increment for the failed element
            new_fext_elem[failed] = -incr_factor * fint_e[failed]

            fext += vector.AssembleNode(new_fext_elem, conn)

            forces_smoothed = False
            for second_iter in range(10):
                ue = vector.AsElement(disp_curr, conn)
                elem0.symGradN_vector(ue, material.F)
                material.F += I2
                material.refresh(compute_tangent=True, element_erosion=True)

                fe = elem.Int_gradN_dot_tensor2_dV(material.Sig)
                fint = vector.AssembleNode(fe, conn)

                Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(material.C)
                K.clear()
                K.assemble(Ke, conn) 
                K.finalize()

                fres_load = fext - fint
                Fext = vector.AsDofs_i(fext)
                Fint = vector.AsDofs_i(fint)
                vector.copy_p(Fint, Fext) 
                nfres = np.sum(np.abs(Fext - Fint))
                nfext = np.sum(np.abs(Fext))
                
                res = nfres / nfext if nfext else nfres

                if res < 1e-06:
                    forces_smoothed = True
                    converged_at_multiplier = incr_factor
                    break

                du.fill(0.0)
                Solver.solve(K, fres_load, du)
                disp_curr += du
                elem.update_x(vector.AsElement(coor + disp_curr, conn))
            
            if forces_smoothed:
                print(f"INFO: Converged at force multiplier {converged_at_multiplier:.2f} for element {failed}.")

                # If converged, update the base state for the next phase or next element
                disp = disp_curr.copy() 
                break # Break from the `force_multipliers_increase` loop

        if not forces_smoothed:
            elem.update_x(vector.AsElement(coor + disp, conn))
            converged_at_multiplier = 1.0

        # Phase 2: Gradually reduce the force if convergence was achieved at a non-zero multiplier
        if converged_at_multiplier is not None and converged_at_multiplier > 0:

            min_steps = 5 
            max_steps = 40 
            
            num_steps_decrease = int(min_steps + (max_steps - min_steps) * converged_at_multiplier)
            num_steps_decrease = max(min_steps, min(max_steps, num_steps_decrease))

            force_multipliers_decrease = np.linspace(converged_at_multiplier, 0.0, num_steps_decrease + 1)[1:] 

            for incr_factor in force_multipliers_decrease:
                new_fext_elem[failed] = -incr_factor * fint_e[failed] 
                
                fext = fext_init.copy() + vector.AssembleNode(new_fext_elem, conn)

                forces_smoothed = False
                for second_iter in range(50):
                    ue = vector.AsElement(disp, conn)
                    elem0.symGradN_vector(ue, material.F)
                    material.F += I2
                    material.refresh(compute_tangent=True, element_erosion=True)

                    fe = elem.Int_gradN_dot_tensor2_dV(material.Sig)
                    fint = vector.AssembleNode(fe, conn)

                    Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(material.C)
                    K.clear()
                    K.assemble(Ke, conn)
                    K.finalize() 

                    fres_load = fext - fint
                    Fext = vector.AsDofs_i(fext)
                    Fint = vector.AsDofs_i(fint)
                    vector.copy_p(Fint, Fext)
                    nfres = np.sum(np.abs(Fext - Fint))
                    nfext = np.sum(np.abs(Fext))
                    
                    res = nfres / nfext if nfext else nfres

                    if res < 1e-06:
                        forces_smoothed = True
                        # disp_curr already holds the converged state, no need to copy
                        break

                    du.fill(0.0)
                    Solver.solve(K, fres_load, du)
                    disp += du
                    elem.update_x(vector.AsElement(coor + disp, conn))
                
                if not forces_smoothed:
                    raise RuntimeError(f"Element erosion for element {failed} unsuccessful.")
                    break
                
    return disp, elem, material

def element_erosion(Solver, vector, conn, material, damage_prev, elem, fint_e, fext_init, disp, newly_failed, K, fe, I2):

    du = np.zeros_like(disp)
        
    # Create new external force vector per element
    new_fext_elem = np.zeros_like(fe)

    # Iterate over each newly failed element
    for failed in newly_failed:
        material.delete_element(failed)
        material.D_damage = damage_prev
 
        # if not forces_smoothed:
        #     #elem.update_x(vector.AsElement(coor + disp, conn))
        converged_at_multiplier = 1.0

        # converged_at_multiplier = 1.0
        initial_guess = np.zeros_like(disp)

        # Phase 2: Gradually reduce the force if convergence was achieved at a non-zero multiplier
        if converged_at_multiplier is not None and converged_at_multiplier > 0:
            print(f"INFO: Reintregating residual external force vector at free nodes of deleted element.")
            min_steps = 11 
            max_steps = 11 
            
            num_steps_decrease = int(min_steps + (max_steps - min_steps) * converged_at_multiplier)
            num_steps_decrease = max(min_steps, min(max_steps, num_steps_decrease))

            force_multipliers_decrease = np.linspace(converged_at_multiplier, 0.0, num_steps_decrease + 1)[1:] 

            for incr_factor in force_multipliers_decrease:
                new_fext_elem[failed] = -incr_factor * fint_e[failed] 
                
                fext = fext_init.copy() + vector.AssembleNode(new_fext_elem, conn)

                forces_smoothed = False
                
                material.increment()

                disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + vector.AsDofs_u(initial_guess), vector.AsDofs_p(disp))

                total_increment = initial_guess.copy()
                for second_iter in range(80):
                    ue = vector.AsElement(disp, conn)
                    elem.symGradN_vector(ue, material.F)
                    material.F += I2
                    material.refresh()

                    fe = elem.Int_gradN_dot_tensor2_dV(material.Sig)
                    fint = vector.AssembleNode(fe, conn)

                    Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(material.C)
                    K.clear()
                    K.assemble(Ke, conn)
                    K.finalize(stabilize=True)

                    fres = fext - fint

                    fres_u = vector.AsDofs_u(fext) - vector.AsDofs_u(fint)
                
                    res_norm = np.linalg.norm(fres_u) 
    
                    if res_norm < 1e-07:
                        forces_smoothed = True
                        print(f"INFO: Increment {incr_factor} converged at Iter {second_iter}, Residual = {res_norm}")
                        # disp_curr already holds the converged state, no need to copy
                        break
                    
                    du.fill(0.0)
                    Solver.solve(K, fres, du)
                    total_increment += du
                    disp += du
                    # elem.update_x(vector.AsElement(coor + disp, conn))
                
                if forces_smoothed:
                    initial_guess = 0.0 * total_increment
                if not forces_smoothed:
                    raise RuntimeError(f"Element erosion for element {failed} unsuccessful.")
                    break
                
    return disp, elem, material

def element_erosion_3D_PBC(Solver, vector, conn, mat, damage_prev, elem, elem0, fint_e, fext_init, disp, newly_failed, K, fe, I2, coor):

    du = np.zeros_like(disp)
        
    # Create new external force vector per element
    new_fext_elem = np.zeros_like(fe)

    # Iterate over each newly failed element
    for failed in newly_failed:
        mat.delete_element(failed)
        mat.D_damage = damage_prev
 

        # converged_at_multiplier = 1.0
        initial_guess = np.zeros_like(disp)

        # Phase 2: Gradually reduce the force if convergence was achieved at a non-zero multiplier
        print(f"INFO: Reintregating residual external force vector at free nodes of deleted element.")
        num_steps_decrease = 11

        force_multipliers_decrease = np.linspace(1.0, 0.0, num_steps_decrease + 1)[1:] 

        for incr_factor in force_multipliers_decrease:
            new_fext_elem[failed] = -incr_factor * fint_e[failed] 
            
            fext = fext_init.copy() + vector.AssembleNode(new_fext_elem, conn)

            forces_smoothed = False
            
            mat.increment()

            disp += initial_guess
            total_increment = initial_guess.copy()
            for iter in range(80):
                ue = vector.AsElement(disp, conn)
                elem0.symGradN_vector(ue, mat.F)
                mat.F += I2
                mat.refresh()

                fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
                fint = vector.AssembleNode(fe, conn)

                Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
                K.clear()
                K.assemble(Ke, conn)
                K.finalize(stabilize=True)

                fres = fext - fint

                if iter > 0:
                    Fext = vector.AsDofs_i(fext)
                    Fint = vector.AsDofs_i(fint)
                    vector.copy_p(Fint, Fext)
                    nfres = np.sum(np.abs(Fext - Fint))
                    nfext = np.sum(np.abs(Fext))
                    if nfext:
                        res = nfres / nfext
                    else:
                        res = nfres

                    if res < 1e-07:
                        forces_smoothed = True
                        print(f"INFO: Increment {incr_factor} converged at Iter {iter}, Residual = {res}")
                        # disp_curr already holds the converged state, no need to copy
                        break
                
                du.fill(0.0)

                # solve
                Solver.solve(K, fres, du)

                total_increment += du
                disp += du
                elem.update_x(vector.AsElement(coor + disp, conn))
            
            if forces_smoothed:
                initial_guess = 0.0 * total_increment
            if not forces_smoothed:
                raise RuntimeError(f"Element erosion for element {failed} unsuccessful.")
                
    return disp, elem, mat


def element_erosion_UL(Solver, vector, coor, conn, material, elem, elem0, fint_e, fext_init, disp, newly_failed, K, fe, I2):

    du = np.zeros_like(disp)
        
    # Create new external force vector per element
    new_fext_elem = np.zeros_like(fe)

    # Iterate over each newly failed element
    for failed in newly_failed:
        print(f"Deleting bulk element {failed}")
        material.delete_element(failed)

        converged_at_multiplier = 1.0
        initial_guess = np.zeros_like(disp)

        print(f"INFO: Reintregating residual external force vector at free nodes of deleted element.")
        min_steps = 5 
        max_steps = 21 
        if failed == 18:
            max_steps = 201
        
        num_steps_decrease = int(min_steps + (max_steps - min_steps) * converged_at_multiplier)
        num_steps_decrease = max(min_steps, min(max_steps, num_steps_decrease))

        force_multipliers_decrease = np.linspace(converged_at_multiplier, 0.0, num_steps_decrease + 1)[1:] 

        for incr_factor in force_multipliers_decrease:

            new_fext_elem[failed] = -incr_factor * fint_e[failed] 
            fext = fext_init.copy() + vector.AssembleNode(new_fext_elem, conn)

            forces_smoothed = False
            
            material.increment()

            total_increment = initial_guess.copy()
            for iter in range(50):
                ue = vector.AsElement(disp, conn)
                elem0.symGradN_vector(ue, material.F)
                material.F += I2
                material.refresh()

                fe = elem.Int_gradN_dot_tensor2_dV(material.Sig)
                fint = vector.AssembleNode(fe, conn)

                Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(material.C)
                K.clear()
                K.assemble(Ke, conn)
                K.finalize(stabilize=True)

                fres = fext - fint

                if iter > 0:
                    fres_u = vector.AsDofs_u(fext) - vector.AsDofs_u(fint)
                
                    res_norm = np.linalg.norm(fres_u) 

                    if res_norm < 1e-06:
                        forces_smoothed = True
                        print(f"INFO: Increment {incr_factor} converged at Iter {iter}, Residual = {res_norm}")
                        # disp_curr already holds the converged state, no need to copy
                        break
                
                du.fill(0.0)
                Solver.solve(K, fres, du)

                disp += du

                elem.update_x(vector.AsElement(coor + disp, conn))
            
            if forces_smoothed:
                continue
            if not forces_smoothed:
                raise RuntimeError(f"Element erosion for element {failed} unsuccessful.")
                break
                
    return disp, elem, material

def element_erosion_multiplemat(Solver, vector, conn, material, damage_prev, elem, fint_e, fext_init, disp, failed, K, fe, I2, mat_failed):

    du = np.zeros_like(disp)
        
    # Create new external force vector per element
    new_fext_elem = np.zeros_like(fe[mat_failed])

    # Iterate over each newly failed element
    material[mat_failed].delete_element(failed)
    for i in range(len(material)):
        material[i].D_damage = damage_prev[i]
 
    # if not forces_smoothed:
    #     #elem.update_x(vector.AsElement(coor + disp, conn))
    converged_at_multiplier = 1.0

    # converged_at_multiplier = 1.0
    initial_guess = np.zeros_like(disp)

    ue = np.empty(len(conn), dtype=object)
    Ke = np.empty(len(conn), dtype=object)
    # Phase 2: Gradually reduce the force if convergence was achieved at a non-zero multiplier
    if converged_at_multiplier is not None and converged_at_multiplier > 0:
        print(f"INFO: Reintregating residual external force vector at free nodes of deleted element.")
        min_steps = 11 
        max_steps = 11 
        
        num_steps_decrease = int(min_steps + (max_steps - min_steps) * converged_at_multiplier)
        num_steps_decrease = max(min_steps, min(max_steps, num_steps_decrease))

        force_multipliers_decrease = np.linspace(converged_at_multiplier, 0.0, num_steps_decrease + 1)[1:] 

        for incr_factor in force_multipliers_decrease:
            new_fext_elem[failed] = -incr_factor * fint_e[mat_failed][failed] 
            
            fext = fext_init.copy() + vector.AssembleNode(new_fext_elem, conn[mat_failed])

            forces_smoothed = False
            
            for mat in material:
                mat.increment()

            disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + vector.AsDofs_u(initial_guess), vector.AsDofs_p(disp))

            total_increment = initial_guess.copy()
            for second_iter in range(80):
                for connectivity, i in enumerate(conn):
                    ue[i] = vector.AsElement(disp, connectivity)
                for u_elem, i in enumerate(ue):
                    elem[i].symGradN_vector(u_elem, material[i].F)
                for tensor,i in enumerate(I2):
                    material[i].F += tensor
                    material[i].refresh()

                for elem_f, i in enumerate(fe):
                    elem_f = elem[i].Int_gradN_dot_tensor2_dV(material[i].Sig)
                    fint += vector.AssembleNode(elem_f, conn[i])

                for element,i in enumerate(elem):
                    Ke[i] = element[i].Int_gradN_dot_tensor4_dot_gradNT_dV(material[i].C)
                K.clear()
                for K_elem,i in enumerate(Ke):
                    K.assemble(K_elem, conn[i])
                K.finalize(stabilize=True)

                fres = fext - fint

                fres_u = vector.AsDofs_u(fext) - vector.AsDofs_u(fint)
            
                res_norm = np.linalg.norm(fres_u) 

                if res_norm < 1e-07:
                    forces_smoothed = True
                    print(f"INFO: Increment {incr_factor} converged at Iter {second_iter}, Residual = {res_norm}")
                    # disp_curr already holds the converged state, no need to copy
                    break
                
                du.fill(0.0)
                Solver.solve(K, fres, du)
                total_increment += du
                disp += du
                # elem.update_x(vector.AsElement(coor + disp, conn))
            
            if forces_smoothed:
                initial_guess = 0.0 * total_increment
            if not forces_smoothed:
                raise RuntimeError(f"Element erosion for element {failed} unsuccessful.")
                break
                
    return disp, elem, material

def element_erosion_3D_PBC_multimat(Solver, vector, conn, mat, damage_prev, elem, elem0, fint_e, fext_init, disp, failed, K, fe, I2, coor, mat_failed):

    du = np.zeros_like(disp)
        
    # Create new external force vector per element
    new_fext_elem = np.zeros_like(fe[mat_failed])

    # Iterate over each newly failed element
    mat[mat_failed].delete_element(failed)
    for i in range(len(mat)):
        mat[i].D_damage = damage_prev[i]
 
    # if not forces_smoothed:
    #     #elem.update_x(vector.AsElement(coor + disp, conn))
    converged_at_multiplier = 1.0

    # converged_at_multiplier = 1.0
    initial_guess = np.zeros_like(disp)

    ue = np.empty(len(conn), dtype=object)
    Ke = np.empty(len(conn), dtype=object)

    # Phase 2: Gradually reduce the force if convergence was achieved at a non-zero multiplier
    print(f"INFO: Reintregating residual external force vector at free nodes of deleted element.")
    num_steps_decrease = 4

    force_multipliers_decrease = np.linspace(1.0, 0.0, num_steps_decrease + 1)[1:] 
    
    for incr_factor in force_multipliers_decrease:
        new_fext_elem[failed] = -incr_factor * fint_e[mat_failed][failed] 
        
        fext = fext_init.copy() + vector.AssembleNode(new_fext_elem, conn[mat_failed])

        forces_smoothed = False
        
        for material in mat:
            material.increment()

        disp += initial_guess
        total_increment = initial_guess.copy()
        for iter in range(20):
            K.clear()
            for i in range(len(mat)):
                ue[i] = vector.AsElement(disp, conn[i]) 
                elem0[i].symGradN_vector(ue[i], mat[i].F)
                mat[i].F += I2[i]
                mat[i].refresh()
                fe[i] = elem[i].Int_gradN_dot_tensor2_dV(mat[i].Sig)
                Ke[i] = elem[i].Int_gradN_dot_tensor4_dot_gradNT_dV(mat[i].C) 
                K.assemble(Ke[i], conn[i]) 
            K.finalize(stabilize=True)

            fint = vector.AssembleNode(fe[0], conn[0])
            for i in range(1,len(mat)):
                fint += vector.AssembleNode(fe[i], conn[i])

            fres = fext - fint

            if iter > 0:
                Fext = vector.AsDofs_i(fext)
                Fint = vector.AsDofs_i(fint)
                vector.copy_p(Fint, Fext)
                nfres = np.sum(np.abs(Fext - Fint))
                nfext = np.sum(np.abs(Fext))
                if nfext:
                    res = nfres / nfext
                else:
                    res = nfres

                if res < 1e-06:
                    forces_smoothed = True
                    print(f"INFO: Increment {incr_factor} converged at Iter {iter}, Residual = {res}")
                    # disp_curr already holds the converged state, no need to copy
                    break
            
            du.fill(0.0)

            # solve
            Solver.solve(K, fres, du)

            total_increment += du
            disp += du
            for i in range(len(mat)):
                elem[i].update_x(vector.AsElement(coor + disp, conn[i]))
        
        if forces_smoothed:
            initial_guess = 0.0 * total_increment
        if not forces_smoothed:
            raise RuntimeError(f"Element erosion for element {failed} unsuccessful.")
                
    return disp, elem, mat

def trial_erosion_3D_PBC_multimat(Solver, vector, conn, mat, damage_prev, elem, elem0, fint_e, fext_base, disp, failed, K, fe, I2, coor, mat_failed):

    du = np.zeros_like(disp)
        
    # Create new external force vector per element
    new_fext_elem = np.zeros_like(fe[mat_failed])

    # Iterate over each newly failed element
    mat[mat_failed].delete_element(failed)

    # converged_at_multiplier = 1.0
    initial_guess = np.zeros_like(disp)

    ue = np.empty(len(conn), dtype=object)
    Ke = np.empty(len(conn), dtype=object)

    # Phase 2: Gradually reduce the force if convergence was achieved at a non-zero multiplier
    print(f"INFO: Reintregating residual external force vector at free nodes of deleted element.")
    num_steps_decrease = 1
    
    forces_smoothed = False
    while not forces_smoothed and num_steps_decrease < 16:
        num_steps_decrease *= 2
        print(f"Attempting to smooth with {num_steps_decrease} steps")
        force_multipliers_decrease = np.linspace(1.0, 0.0, num_steps_decrease+1)[1:] 
        trial_disp = disp.copy()
        for i in range(len(mat)):
            elem[i].update_x(vector.AsElement(coor + trial_disp, conn[i]))
            mat[i].D_damage = damage_prev[i]
        for incr_factor in force_multipliers_decrease:
            new_fext_elem[failed] = -incr_factor * fint_e[mat_failed][failed] 
            
            fext = fext_base.copy() + vector.AssembleNode(new_fext_elem, conn[mat_failed])

            forces_smoothed = False

            for iter in range(12):
                K.clear()
                for i in range(len(mat)):
                    ue[i] = vector.AsElement(trial_disp, conn[i]) 
                    elem0[i].symGradN_vector(ue[i], mat[i].F)
                    mat[i].F += I2[i]
                    mat[i].refresh()
                    fe[i] = elem[i].Int_gradN_dot_tensor2_dV(mat[i].Sig)
                    Ke[i] = elem[i].Int_gradN_dot_tensor4_dot_gradNT_dV(mat[i].C) 
                    K.assemble(Ke[i], conn[i]) 
                K.finalize(stabilize=True)

                fint = vector.AssembleNode(fe[0], conn[0])
                for i in range(1,len(mat)):
                    fint += vector.AssembleNode(fe[i], conn[i])

                fres = fext - fint

                if iter > 0:
                    Fext = vector.AsDofs_i(fext)
                    Fint = vector.AsDofs_i(fint)
                    vector.copy_p(Fint, Fext)
                    nfres = np.sum(np.abs(Fext - Fint))
                    nfext = np.sum(np.abs(Fext))
                    if nfext:
                        res = nfres / nfext
                    else:
                        res = nfres

                    if res < 1e-06:
                        forces_smoothed = True
                        print(f"INFO: Increment {incr_factor} converged at Iter {iter}, Residual = {res}")
                        # disp_curr already holds the converged state, no need to copy
                        break
                
                du.fill(0.0)

                # solve
                Solver.solve(K, fres, du)

                trial_disp += du
                for i in range(len(mat)):
                    elem[i].update_x(vector.AsElement(coor + trial_disp, conn[i]))
    
    if forces_smoothed:
        disp = trial_disp
    if not forces_smoothed:
        raise RuntimeError(f"Element erosion for element {failed} unsuccessful.")
                
    return disp, elem, mat