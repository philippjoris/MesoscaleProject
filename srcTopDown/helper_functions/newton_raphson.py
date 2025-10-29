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


def newton_raphson_single(
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
        ue = vector.AsElement(disp, conn) 
        elem0.symGradN_vector(ue, mat.F)
        mat.F += I2
        mat.refresh()
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C) 
        K.assemble(Ke, conn) 
        K.finalize(stabilize=True)

        fint = vector.AssembleNode(fe, conn)

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
        elem.update_x(vector.AsElement(coor + disp, conn))

    return disp, elem, mat, converged, total_increment, res, iter