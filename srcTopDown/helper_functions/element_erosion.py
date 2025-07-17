"""
file: element_erosion.py
author: Philipp van der Loos

A helper function for element erosion.

"""

import numpy as np

def element_erosion(Solver, vector, coor, material, elem, elem0, fint_e, fext_init, disp, newly_failed, K, fe, I2, meshRefined):

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
            elem.update_x(vector.AsElement(coor + disp ))

            # Apply the current force increment for the failed element
            new_fext_elem[failed] = -incr_factor * fint_e[failed]

            fext += vector.AssembleNode(new_fext_elem)

            forces_smoothed = False
            for second_iter in range(10):
                ue = vector.AsElement(disp_curr)
                elem0.symGradN_vector(ue, material.F)
                material.F += I2
                material.refresh(compute_tangent=True, element_erosion=True)

                fe = elem.Int_gradN_dot_tensor2_dV(material.Sig)
                fint = vector.AssembleNode(fe)

                Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(material.C)
                K.assemble(Ke) 

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
                elem.update_x(vector.AsElement(coor + disp_curr))
            
            if forces_smoothed:
                print(f"INFO: Converged at force multiplier {converged_at_multiplier:.2f} for element {failed}.")

                # If converged, update the base state for the next phase or next element
                disp = disp_curr.copy() 
                break # Break from the `force_multipliers_increase` loop

        if not forces_smoothed:
            elem.update_x(vector.AsElement(coor + disp ))
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
                
                fext = fext_init.copy() + vector.AssembleNode(new_fext_elem)

                forces_smoothed = False
                for second_iter in range(50):
                    ue = vector.AsElement(disp)
                    elem0.symGradN_vector(ue, material.F)
                    material.F += I2
                    material.refresh(compute_tangent=True, element_erosion=True)

                    fe = elem.Int_gradN_dot_tensor2_dV(material.Sig)
                    fint = vector.AssembleNode(fe)

                    Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(material.C)
                    K.assemble(Ke)

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
                    elem.update_x(vector.AsElement(coor + disp))
                
                if not forces_smoothed:
                    raise RuntimeError(f"Element erosion for element {failed} unsuccessful.")
                    break
                
    return disp, elem, material