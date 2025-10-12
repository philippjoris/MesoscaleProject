import numpy as np

def nodesPeriodic3D(mesh):
    """
    Recreates the C++ function for 3D hierarchical periodic node pairings 
    on a cube (RVE). FBL corner is the master.
    
    Args: (All edge/face arrays must be consistently sorted for pairing)
        fro, bck, ...: 1D NumPy arrays of node indices for open faces/edges.
        corner_...: Single integer index for corner nodes.
    
    Returns:
        np.ndarray: A 2D array of node pairings (master_node, slave_node).
    """
    # Open Face Nodes (excluding edges/corners)
    fro = mesh['face_fro']
    bck = mesh['face_bck']
    lft = mesh['face_lft']
    rgt = mesh['face_rgt']
    bot = mesh['face_bot']
    top = mesh['face_top']
    # Open Edge Nodes (excluding corners)
    froBot = mesh['edge_froBot']
    froTop = mesh['edge_froTop']
    froLft = mesh['edge_froLft']
    froRgt = mesh['edge_froRgt']
    bckBot = mesh['edge_bckBot']
    bckTop = mesh['edge_bckTop']
    bckLft = mesh['edge_bckLft']
    bckRgt = mesh['edge_bckRgt']
    botLft = mesh['edge_botLft']
    botRgt = mesh['edge_botRgt']
    topLft = mesh['edge_topLft']
    topRgt = mesh['edge_topRgt']
    # Corner Nodes (single indices)
    corner_FBL = mesh['corner_froBotLft']
    corner_FBR = mesh['corner_froBotRgt']
    corner_BBR = mesh['corner_bckBotRgt']
    corner_BBL = mesh['corner_bckBotLft']
    corner_FTL = mesh['corner_froTopLft']
    corner_FTR = mesh['corner_froTopRgt']
    corner_BTR = mesh['corner_bckTopRgt']
    corner_BTL = mesh['corner_bckTopLft']
    # --- 1. Calculate Array Size (Matching C++ Logic) ---
    
    # Total Face Pairs (Face <-> Face)
    tface = fro.size + lft.size + bot.size 
    
    # Total Edge Pairs (Edge <-> Edge)
    # The C++ code uses 3 pairings per master edge:
    # 3 * froBot.size() (froBot <-> bckBot, froBot <-> bckTop, froBot <-> froTop)
    # 3 * froLft.size() (froLft <-> froRgt, froLft <-> bckRgt, froLft <-> bckLft)
    # 3 * botLft.size() (botLft <-> botRgt, botLft <-> topRgt, botLft <-> topLft)
    tedge = 3 * froBot.size + 3 * froLft.size + 3 * botLft.size
    
    # Total Corner Pairs (Corner <-> Corner)
    tnode = 7 # FBL is master, 7 slaves
    
    num_rows = tface + tedge + tnode
    
    # Initialize the result array (size_t is usually 64-bit int, use np.intp or np.uint64)
    ret = np.empty((num_rows, 2), dtype=np.intp)
    i = 0

    # --- 2. Corner Node Pairings (7 Pairs) ---
    # Master node: Front-Bottom-Left (FBL)
    
    # i=0: FBL -> FBR (X)
    ret[i, 0], ret[i, 1] = corner_FBL, corner_FBR; i += 1
    # i=1: FBL -> BBR (Diagonal-X/Z)
    ret[i, 0], ret[i, 1] = corner_FBL, corner_BBR; i += 1
    # i=2: FBL -> BBL (Z)
    ret[i, 0], ret[i, 1] = corner_FBL, corner_BBL; i += 1
    # i=3: FBL -> FTL (Y)
    ret[i, 0], ret[i, 1] = corner_FBL, corner_FTL; i += 1
    # i=4: FBL -> FTR (Diagonal-X/Y)
    ret[i, 0], ret[i, 1] = corner_FBL, corner_FTR; i += 1
    # i=5: FBL -> BTR (Diagonal-X/Y/Z)
    ret[i, 0], ret[i, 1] = corner_FBL, corner_BTR; i += 1
    # i=6: FBL -> BTL (Diagonal-Y/Z)
    ret[i, 0], ret[i, 1] = corner_FBL, corner_BTL; i += 1

    # --- 3. Open Edge Pairings (9 Master Edges $\times$ 3 Pairings) ---
    
    # Master Edge Set 1: Front-Bottom (froBot)
    # 3 * froBot.size pairs (X-direction edge, periodic in Y and Z)
    
    # Z-periodicity: froBot (master) <-> bckBot (slave 1)
    ret[i : i + froBot.size, 0] = froBot
    ret[i : i + froBot.size, 1] = bckBot
    i += froBot.size

    # Diagonal periodicity: froBot (master) <-> bckTop (slave 2)
    ret[i : i + froBot.size, 0] = froBot
    ret[i : i + froBot.size, 1] = bckTop
    i += froBot.size

    # Y-periodicity: froBot (master) <-> froTop (slave 3)
    ret[i : i + froBot.size, 0] = froBot
    ret[i : i + froBot.size, 1] = froTop
    i += froBot.size

    # Master Edge Set 2: Bottom-Left (botLft)
    # 3 * botLft.size pairs (Z-direction edge, periodic in X and Y)

    # X-periodicity: botLft (master) <-> botRgt (slave 1)
    ret[i : i + botLft.size, 0] = botLft
    ret[i : i + botLft.size, 1] = botRgt
    i += botLft.size

    # Diagonal periodicity: botLft (master) <-> topRgt (slave 2)
    ret[i : i + botLft.size, 0] = botLft
    ret[i : i + botLft.size, 1] = topRgt
    i += botLft.size

    # Y-periodicity: botLft (master) <-> topLft (slave 3)
    ret[i : i + botLft.size, 0] = botLft
    ret[i : i + botLft.size, 1] = topLft
    i += botLft.size

    # Master Edge Set 3: Front-Left (froLft)
    # 3 * froLft.size pairs (Y-direction edge, periodic in X and Z)

    # X-periodicity: froLft (master) <-> froRgt (slave 1)
    ret[i : i + froLft.size, 0] = froLft
    ret[i : i + froLft.size, 1] = froRgt
    i += froLft.size

    # Diagonal periodicity: froLft (master) <-> bckRgt (slave 2)
    ret[i : i + froLft.size, 0] = froLft
    ret[i : i + froLft.size, 1] = bckRgt
    i += froLft.size

    # Z-periodicity: froLft (master) <-> bckLft (slave 3)
    ret[i : i + froLft.size, 0] = froLft
    ret[i : i + froLft.size, 1] = bckLft
    i += froLft.size
    
    # --- 4. Open Face Pairings (3 Pairs) ---
    
    # Z-periodicity: Front (fro, master) <-> Back (bck, slave)
    ret[i : i + fro.size, 0] = fro
    ret[i : i + fro.size, 1] = bck
    i += fro.size

    # X-periodicity: Left (lft, master) <-> Right (rgt, slave)
    ret[i : i + lft.size, 0] = lft
    ret[i : i + lft.size, 1] = rgt
    i += lft.size

    # Y-periodicity: Bottom (bot, master) <-> Top (top, slave)
    ret[i : i + bot.size, 0] = bot
    ret[i : i + bot.size, 1] = top
    i += bot.size

    # Check to ensure all rows were filled
    # assert i == num_rows, "Mismatch in calculated array size and filled rows."

    return ret