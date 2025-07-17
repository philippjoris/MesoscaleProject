"""

UNDER CONSTRUCTION - NOT USABLE YET

A helper function for element erosion.

# ------------------------
# ------------------------
# build a check that only the NEWLY failed elements are 
# dealt with!
# ------------------------
# ------------------------

"""

import numpy as np

def element_erosion(Solver, vector, coor, material, elem, elem0, fint_e, fext, disp, failed_elem, K, fe, I2):
    du = np.zeros_like(disp)
    fext_curr = fext.copy()
    fe_curr = fint_e.copy()

    curr_failed = failed_elem.copy()
    # delete cells with damage > 1 and append to vector
    for k, obj in enumerate(material.D_damage):
        if any(IP>=1 for IP in obj):
            curr_failed.add(k)

    newly_failed = curr_failed - failed_elem
    if len(newly_failed) > 2:
        a = 1
    if newly_failed:
        print(f"INFO: Elements {newly_failed}  failed.")
        # create new external force vector per element
        new_fext_elem = np.zeros_like(fe)

        for failed in newly_failed:
            material.delete_element(failed)                    
            for incr in np.linspace(1.0, 0.0, 20):
                # apply internal forces to external forces 
                new_fext_elem[failed] = -incr * fe_curr[failed]                                        

                # assemble force vector
                fext = fext_curr + vector.AssembleNode(new_fext_elem)
                                
                forces_smoothed = False
                
                for second_iter in range(50):
                    ue = vector.AsElement(disp)
                    elem0.symGradN_vector(ue, material.F)
                    material.F += I2
                    material.refresh(compute_tangent=True, element_erosion=True) 

                    # internal force
                    fe = elem.Int_gradN_dot_tensor2_dV(material.Sig)
                    
                    fint = vector.AssembleNode(fe)

                    # stiffness matrix
                    Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(material.C)
                    K.assemble(Ke)

                    fres_load = fext - fint
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
                        break

                    du.fill(0.0)

                    Solver.solve(K, fres_load, du)
                    
                    disp += du    
                    
                    elem.update_x(vector.AsElement(coor + disp))

                if not forces_smoothed:
                    raise RuntimeError(f"Free forces due to element erosion not smoothed successfully.")      
            
        return disp, elem, material, curr_failed
    else:
        return disp, elem, material, curr_failed