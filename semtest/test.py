from sem import *

import numpy as np
import pandas as pd
import cProfile

if __name__=="__main__":
    np.random.seed(7)
    dt = 0.001
    s = Model(ne=256, d=2)

    # load spatial data
    grid = np.loadtxt("../input_data/SpatialRef/pts.txt")
    expr_data = pd.read_csv("../input_data/SpatialRef/spatial_expr_normalized.csv", index_col=0)

    # set initial SEM to match the reference data exactly
    s.load_from_data()
    #voronoi_plot(s, pause=False, gene_color=False)
    s.sem_simulation(nsteps=1000, dt=dt, division=False, transition=False, dot=False)
    #voronoi_plot(s, pause=False, gene_color=False)

    #s.compute_gene_profiles()

    pr = cProfile.Profile()
    pr.enable()

    # each cycle consists of cell lifecycle, OT force, and SEM force
    cycles = 1
    for i in range(cycles):
        print("Iteration %3d\tNumber of active cells: %d"%(i, len(s.vact)), end='\n')
        for ii in range(100):
            #s.sem_simulation(nsteps=100, dt=dt, division=True, transition=True, dot=["FLG","CALML5","LOR","SPINK5", "MLANA", "STMN1"])
            s.sem_simulation(nsteps=100, dt=dt, division=True, transition=True,dot=["LOR"])
            #s.sem_simulation(nsteps=100, dt=dt, division=True, transition=True, dot=False)
        s.print_cell_type_distribution()
        #voronoi_plot(s, pause=True, gene_color=False)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(10)
    s.compute_gene_profiles()
    voronoi_plot(s, pause=False, gene_color=False)