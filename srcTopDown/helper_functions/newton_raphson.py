import numpy as np

def newton_raphson_solve(
    disp,
    initial_guess,
    max_iter,
    control,
    vector,
    conn,
    elem0,
    elem,
    mat,
    ue,
    fe,
    fint,
    Ke,
    K,
    fext,
    Fext,
    Fint,
    Solver,
    du,
    F,
    I2,
    coor,
    RES_TOL=1e-06
):
    """
    Performs the Newton-Raphson iterative solve for one load increment.

    Returns:
        disp (np.array): Updated displacement vector.
        elem (GooseFEM.Element): Updated element object (due to erosion).
        mat (GMat): Updated material state.
        converged (bool): True if convergence was reached.
        total_increment (np.array): Total displacement increment applied.
    """
    converged = False
    
    # mat.increment() is performed *outside* the loop in your original script
    # disp += initial_guess is also performed *outside* the loop
    
    total_increment = initial_guess.copy()
    
    # The actual N-R loop
    for iter in range(max_iter):
        # 1. Kinematics/Material Update
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

        fres = -fint                      
        # 5. Convergence Check
        if iter > 0:
            vector.asDofs_i(fext, Fext)
            vector.asDofs_i(fint, Fint)
            vector.copy_p(Fint, Fext)                                       
        
            nfres = np.sum(np.abs(Fext - Fint))
            nfext = np.sum(np.abs(Fext))
            
            # - relative residual, for convergence check
            if nfext:
                res = nfres / nfext
            else:
                res = nfres
            
            if res < RES_TOL:
                converged = True
                break
                
        
        # 6. Solve and Update
        du.fill(0.0)

        # Initial displacement update (only on first iteration)
        if iter == 0:
             # Calculate the kinematic part of the current increment
             # The F is the total deformation gradient, so (F - I) is the total displacement gradient.
             du[control.controlNodes, 0] = (F[0,:] - np.eye(3)[0, :]) 
             du[control.controlNodes, 1] = (F[1,:] - np.eye(3)[1, :])
             du[control.controlNodes, 2] = (F[2,:] - np.eye(3)[2, :])
             
        # Solve the system
        Solver.solve(K, fres, du)
        
        # Add delta u
        disp += du
        total_increment += du
        
        # Update element coordinates (for large deformation)
        for i in range(len(mat)):
            elem[i].update_x(vector.AsElement(coor + disp, conn[i]))

    return disp, elem, mat, converged, total_increment, res, iter













#if mode is 'load_step':
#                    if (np.amax(mat.D_damage) < 1):
#                        break
#                    else:
#                        curr_failed = failed_elem.copy()
#                        # delete elements with damage > 1 and append to vector
#                        for k, obj in enumerate(mat.D_damage):
#                            if any(IP >= 1 for IP in obj):
#                                curr_failed.add(k)
#                                # break
#
#                        newly_failed = curr_failed - failed_elem
#                        failed_elem = curr_failed.copy()
#
#                        to_be_deleted.extend(list(newly_failed))
#
#                        if to_be_deleted:
#                            print(f"INFO: Element(s) failed: {to_be_deleted}.")
#                            elem_to_delete = {to_be_deleted.pop(0)} 
#                            print( f"INFO: Deleting element {elem_to_delete}.")
#                            mat.delete_element(elem_to_delete[0])
#                            mat.D_damage = damage_prev
#                            decreasing_steps = 11
#                            new_fext = np.zeros_like(fe)
#                            initial_guess = np.zeros_like(disp)
#                            for incr_factor in decreasing_steps:
#                                new_fext[elem_to_delete[0]] = -incr_factor * fe[elem_to_delete[0]]
#                                fext = fext.copy() + vector.AssembleNode(new_fext, conn)    
#                                forces_smoothed = False
#                                disp, elem, mat, forces_smoothed, total_increment = newton_raphson_solve(
#                                        disp,
#                                        initial_guess,
#                                        max_iter,
#                                        periodicity,
#                                        vector,
#                                        conn,
#                                        elem0,
#                                        elem,
#                                        mat,
#                                        damage_prev,
#                                        fe,
#                                        fint,
#                                        K,
#                                        Fext,
#                                        Fint,
#                                        Solver,
#                                        du,
#                                        F,
#                                        I2,
#                                        coor,
#                                        mode='force_step',
#                                        RES_TOL=1e-06
#                                ) 
#                                if not forces_smoothed:
#                                    raise RuntimeError(f"Element erosion for element {elem_to_delete} unsuccessful.")     