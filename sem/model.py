import math
import numba
import torch
import pickle
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import numpy as np
import math
import pandas as pd
from scipy.spatial import distance_matrix, Delaunay
import seaborn as sns

from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, MovieWriter

import cProfile, pstats

from dot.dot import gaussian_diffusion, sinkhorn_knopp, sinkhorn_stabilized, sinkhorn_epsilon_scaling

from visualization import *
from sem import *

# be able to adaptively run on CPU or GPU
try_gpu = True
device = torch.device('cuda:0' if (try_gpu and torch.cuda.is_available()) else 'cpu')


# TODO: describe class
class Model:
    def __init__(self, ne=100, d=3):
        self.d = d
        self.ne = ne  # number of elements
        self.xe = np.empty([self.ne, self.d], np.float32)  # coordinates of elements
        self.ecid = np.empty([self.ne], np.int32)  # cell id of elements
        self.etyp = np.empty([self.ne], np.int32)  # cell type of elements
        self.eact = np.ones([self.ne], np.bool8)  # whether or not each element is currently active
        if self.d != 2:
            raise ValueError
        self.xe = np.empty(shape=(self.ne, self.d))
        self.xwidth = 26
        self.xe[:, 0] = np.random.uniform(0, self.xwidth, size=self.ne)
        # initialize y coordinate based on x coordinate
        for i in range(self.ne):
            xv = self.xe[:, 0]
            ymin = 3 + 3*np.sin(2*np.pi*xv/26)
            self.xe[:, 1] = np.random.uniform(ymin, 10)
        self.ecid[:] = np.array(list(range(self.ne)))
        self.etyp[:] = 0
        self.d_ecid = cuda.to_device(self.ecid)
        self.d_etyp = cuda.to_device(self.etyp)
        self.d_xe = cuda.to_device(self.xe)
        self.d_xe_F = cuda.to_device(self.xe)
        self.d_eact = cuda.to_device(self.eact)
        self.ceid = {}
        self.update_ceid()
        self.cfeat = {}
        self.ctyp = np.zeros(shape=self.ne, dtype=np.int32)
        self.dot_P1 = {}
        self.dot_b = {}
        self.istep = 0
        # cell IDs that are not currently in use
        self.free_ids = set()

        # set up GPU RNG
        if self.ne > 1024:
            raise RuntimeError
        self.threads_per_block = self.ne
        # TODO: might be bad if this is more than 1
        self.blocks_per_grid = 1
        self.rng_states = create_xoroshiro128p_states(self.threads_per_block * self.blocks_per_grid, seed=1)

        # parameters controlling life cycle model (higher = faster)
        self.lifecycle_timescale = 1

        # parameter controlling strength of dot force
        self.dot_alpha = 10

    def sem_simulation(self, nsteps=1000, dt=0.01, division=True, transition=True, dot=True):
        self.update_etyp()
        if self.d != 2:
            raise NotImplementedError

        if dot is True:
            gene_set = self.gene_set
        elif dot:
            gene_set = dot

        # advance cell life cycle
        #for i in range(nsteps):
        if division:
            self.cell_birth_death(nsteps*dt)
        if transition:
            self.cell_transition(nsteps*dt)

        self.d_xe = cuda.to_device(self.xe)
        self.d_xe_F = cuda.to_device(self.xe)
        cuda.synchronize()
        nact = len(self.vact)
        move_point_2[self.blocks_per_grid, nact](self.d_xe, self.d_xe_F, self.rng_states, self.d_etyp,self.d_vact, 0.5, 1.25, dt, nsteps)
        if dot:
            self.dot_simulation_compute_gradient(gene_set)
        cuda.synchronize()
        self.d_xe.copy_to_host(self.xe)
        if dot:
            self.dot_simulation_apply_gradient()


    def update_etyp(self):
        for k, v in self.ceid.items():
            self.etyp[v] = self.ctyp[k]
        self.d_etyp = cuda.to_device(self.etyp)

    def cell_division(self, cid, type = 1):
        """Divide cell of cid into two.
        """
        # Divide the a cell without adding elements
        if type == 1:
            # split the cell along a random axis
            # TODO: make this a long axis
            vec = np.random.rand(self.d)
            eid = np.where(self.ecid == cid)[0]
            nc = np.max(self.ecid)+1
            dis = self.xe[eid, :].dot(vec.reshape(-1, 1))
            mdis = np.median(dis)
            tmp_eid = np.where(dis <= mdis)[0]
            self.ecid[eid[tmp_eid]] = nc
        self.d_ecid = cuda.to_device(self.ecid)
        self.update_ceid()
        cuda.synchronize()

    def cell_birth_death(self, dt):
        """
        Simulate cell birth and death process for time dt, causing death of cells randomly. Currently as
        :return:
        """

        def death_prob(k, dt):
            """
            Function determining rate of cell death for a cell at height y
            :param k: cell type index
            :param dt: timestep of simulation
            :return: probability of cell death in current timestep
            """
            # TODO: proper exponential distribution or is that too slow?
            # TODO: determine appropriate death timescale
            # mean lifetime of cell
            alpha = 200/self.lifecycle_timescale
            type_scale = self.death_rate_by_type[k]
            # mean lifetime = alpha/type_scale (type_scale may be 0 = infinite lifetime)
            return 1 - np.exp(-type_scale*dt/alpha)
            #return type_scale*dt/alpha

        # precompute death probability for each cell type
        death_probs = np.empty(shape=self.ntypes, dtype=np.float64)
        for i in range(self.ntypes):
            death_probs[i] = death_prob(i, dt)

        xc = self.get_cell_center()
        xr = np.random.rand(len(xc))
        for ind, (i, v) in enumerate(xc.items()):
            y = v[1]
            if xr[ind] < death_probs[self.ctyp[i]]:
                # kill cell i by deactivating its elements
                self.eact[self.ecid == i] = False
                self.ecid[self.ecid == i] = -1
                self.free_ids.add(i)
        self.d_ecid = cuda.to_device(self.ecid)
        self.update_ceid()

        # TODO: process cell division (birth)
        def birth_prob(k, dt, nact, ne):
            """
            Function determining probability a given cell divides
            :param k: index of type of cell
            :param dt: timestep of simulation
            :param nact: number of active cells
            :param ne: number of total elements (possible cells)
            :return: probability of cell division during timestep
            """

            # controls rate of division (bigger = slower division)
            alpha = 2/self.lifecycle_timescale

            # logistic scaling of birth rate down with population
            num_scale = 1 - nact/ne

            scale = num_scale*self.birth_rate_by_type[k]
            if scale < 0:
                scale = 0
            return 1 - np.exp(-scale*dt/alpha)

        # go through all active cells and check for division
        xc = self.get_cell_center()
        nc = len(xc)

        # count current number of basal cells for use in division
        basal_count = np.zeros(shape=(4,))
        for i in xc.keys():
            if self.ctyp[i] <= 3:
                basal_count[self.ctyp[i]] += 1

        for ind, (i, v) in enumerate(xc.items()):
            # TODO: work out dependence of division on cell type
            if nc >= self.ne-1:
                # out of new elements to divide with
                break
            y = v[1]
            x = v[0]
            p = birth_prob(self.ctyp[i], dt, nc, self.ne)
            if p > 0 and np.random.rand() < p:
                # this cell divides
                # get an element for the new cell
                new_element = np.nonzero(self.ecid==-1)[0][0]
                # get an id for the new cell
                new_id = self.free_ids.pop()
                # record the element membership
                self.ecid[new_element] = new_id
                # put the element near the parent
                new_pos = v + np.random.normal(0.0, 0.5, size=v.shape)
                self.xe[new_element, :] = new_pos
                # set gene expression
                #for k in self.cfeat.keys():
                #    self.cfeat[k][new_id] = self.cfeat[k][i]

                # activate element
                self.eact[new_element] = True

                # set cell type using asymmetric probabilities based on number of basal cells
                count = basal_count[self.ctyp[i]]
                if count < 8:
                    pb = 2/3
                elif count > 15:
                    pb = 0.3
                else:
                    pb = 1/3
                pv = [pb, 1/3, 2/3 - pb]
                v = np.random.choice(a=3, p=pv)
                if v == 0:
                    # two basal cells
                    self.ctyp[new_id] = self.ctyp[i]
                elif v == 1:
                    # asymmetric division into spinous cell
                    self.ctyp[new_id] = 4
                elif v == 2:
                    # division into two spinous cells
                    self.ctyp[new_id] = 4
                    self.ctyp[i] = 4

                nc += 1

        self.d_ecid = cuda.to_device(self.ecid)
        self.update_ceid()
        self.update_vact()
        self.update_etyp()

    def update_vact(self):
        self.vact = np.nonzero(self.eact)[0].astype(np.int32)
        self.d_eact = cuda.to_device(self.eact)
        self.d_vact = cuda.to_device(self.vact)

    def update_cell_lifecycle(self, dt):
        # advance all cells forward by a time step dt
        pass


    def cell_transition(self, dt):
        """
        Process transitions between cell types over a timestep dt
        :return:
        """
        # TODO: this factor controls the overall rate of transitions - sort that out
        alpha = 0.001

        # TODO: this is not the right way to do Markov transition matrices from rates
        A = alpha * dt * self.transition_matrix * self.lifecycle_timescale
        rowsum = 1-np.sum(A, axis=1)
        # failsafe cause of the above bad math
        if np.any(rowsum < 0):
            raise RuntimeError
        A = np.diag(rowsum) + A
        for k in self.ceid.keys():
            cur_type = self.ctyp[k]
            # TODO: pretty sure the transition matrix is also transposed from convention
            new_type = np.random.choice(a=self.ntypes, p=A[cur_type, :])
            self.ctyp[k] = new_type

        self.update_etyp()


    def split_cell_elements(self):
        """
        Testing function that splits up elements so that each corresponds to a single cell
        :return:
        """
        for i in range(self.ne):
            self.ecid[i] = i
        self.d_ecid = cuda.to_device(self.ecid)
        self.update_ceid()

    def update_ceid(self):
        # This currently iterates over all possible IDs including those that don't currently correspond to active cells
        nc = np.max(self.ecid) + 1
        self.ceid = {}
        for i in range(nc):
            ceid = np.where(self.ecid == i)[0]
            if len(ceid) > 0:
                self.ceid[i] = ceid

    # TODO: generalize for 2d/3d geometries
    def dot_initialize(self, pts, pts_data, data_id, grids_np, dtype=torch.float32, device=torch.device("cuda")):
        # Set up the grid, respecting periodicity
        #M_np = distance_matrix(grids_np, grids_np)
        npg = grids_np.shape[0]
        M_np = np.zeros(shape=[npg, npg])
        for i in range(npg):
            for j in range(i+1, npg):
                # compute pairwise distances periodic in x
                xdist = np.abs(grids_np[i, 0] - grids_np[j, 0])
                if xdist > self.xwidth/2:
                    xdist = self.xwidth - xdist
                ydist = grids_np[i, 1] - grids_np[j, 1]
                dist = np.sqrt(xdist**2 + ydist**2)
                M_np[i, j] = dist

        # Set up reference data
        npt_1 = pts.shape[0]
        nus_1_np = np.ones(npt_1)
        pts_1 = torch.tensor(pts, dtype=dtype, device=device)
        sigmas_1 = torch.tensor(pts_data[:, 1], dtype=dtype, device=device)
        peaks_1 = torch.tensor(pts_data[:, 0], dtype=dtype, device=device)
        nus_1 = torch.tensor(nus_1_np, dtype=dtype, device=device)

        grids = torch.tensor(grids_np, dtype=dtype, device=device)
        M = torch.tensor(M_np, dtype=dtype, device=device)
        P1 = gaussian_diffusion(pts_1, grids, peaks_1, sigmas_1, nus_1, x_periodic=26)
        self.dot_P1[data_id] = P1
        self.dot_b[data_id] = P1 / torch.sum(P1)
        self.dot_M = M
        self.dot_grids = grids

    def get_cell_center(self):
        xc = {}
        for (k, v) in self.ceid.items():
            xc[k] = np.mean(self.xe[v, :], axis=0)

        return xc

    def get_expr_peaks_sigmas(self, gene, xc=None):
        if not xc:
            xc = self.get_cell_center()
        pts = []
        peaks = []
        sigmas = []
        self.cids = []
        for k, v in xc.items():
            # Get the expression for cell k
            peaks.append(self.type_expr.loc[self.ctyp[k], gene])
            pts.append(xc[k])
            self.cids.append(k)
            sigmas.append(1.0)
        pts = np.array(pts, float)
        peaks = np.array(peaks, float)
        sigmas = np.array(sigmas, float)

        return pts, peaks, sigmas

    def vals_to_grid(self, pts_0_np, peaks=None, sigmas=None, dtype=torch.float32,
                         device=torch.device("cuda")):
        # Get the cell center locations
        npt_0 = pts_0_np.shape[0]
        nus_0_np = np.ones(npt_0)
        pts_0 = torch.tensor(pts_0_np, dtype=dtype, device=device, requires_grad=True)
        if sigmas is None:
            sigmas_0 = torch.ones(npt_0, dtype=dtype, device=device)
        else:
            sigmas_0 = torch.tensor(sigmas, dtype=dtype, device=device)
        if peaks is None:
            peaks_0 = torch.ones(npt_0, dtype=dtype, device=device)
        else:
            peaks_0 = torch.tensor(peaks, dtype=dtype, device=device)
        nus_0 = torch.tensor(nus_0_np, dtype=dtype, device=device)
        P0 = gaussian_diffusion(pts_0, self.dot_grids, peaks_0, sigmas_0, nus_0, x_periodic=26)
        #a = P0 / torch.sum(P0)
        return self.dot_grids, pts_0, P0

    def dot_get_gradient(self, pts_0_np, data_id, peaks=None, sigmas=None, dtype=torch.float32,
                         device=torch.device("cuda")):
        _, pts_0, P0 = self.vals_to_grid(pts_0_np, peaks=peaks, sigmas=sigmas, dtype=dtype,
                         device=device)
        a = P0 / torch.sum(P0)
        D, _ = sinkhorn_knopp(a, self.dot_b[data_id], self.dot_M, 1.0)
        D.backward()
        with torch.no_grad():
            tmp_grad = pts_0.grad
            # pts_0.grad.detach_()
            # pts_0.grad.zero_()
            # print(tmp_grad)

        return tmp_grad.cpu().numpy()

    def dot_simulation_compute_gradient(self, genes=None):
        xc = self.get_cell_center()
        nc = len(xc)
        self.pts_grad = np.zeros(shape=[nc, 2])

        # default behavior is to use all possible genes
        if not genes:
            genes = self.gene_set

        for gene in genes:
            pts, peaks, sigmas = self.get_expr_peaks_sigmas(gene, xc=xc)
            self.pts_grad += self.dot_get_gradient(pts, gene, peaks=peaks, sigmas=sigmas, device=device)
        return

    def dot_simulation_apply_gradient(self):
        for icell in range(len(self.cids)):
            cid = self.cids[icell]
            ceid = self.ceid[cid]
            self.xe[ceid] -= self.dot_alpha*self.pts_grad[icell, :].reshape(1, -1)
        self.d_xe = cuda.to_device(self.xe)

    def compute_gene_profiles(self):
        xc = self.get_cell_center()
        for gene in self.gene_set:
            # get the expression grid for this gene
            pts, peaks, sigmas = self.get_expr_peaks_sigmas(gene, xc=xc)
            sigmas = 3*sigmas
            grid_pts, _, a = self.vals_to_grid(pts, peaks=peaks, sigmas=sigmas)
            grid_pts = grid_pts.cpu().detach().numpy()
            a = a.cpu().detach().numpy()
            plot_gene_profile(grid_pts, a, self.expr_data.loc[:, gene], gene)

    def print_cell_type_distribution(self):
        type_names = ["BAS-1", "BAS-2", "BAS-3", "BAS-4", "SPN-1", "SPN-2", "GRN"]
        types = self.ctyp[list(self.ceid.keys())]
        for i in range(len(type_names)):
            print(type_names[i], "%.3f"%(np.count_nonzero(types==i)/len(types)), end=' ')
        print("\n")


    def save_pos(self, fn):
        np.savez(fn, xe=self.xe, ecid=self.ecid, ceid=self.ceid)

    def load_pos(self, fn):
        data = np.load(fn, allow_pickle=True)
        self.xe = data['xe']
        self.ecid = data['ecid']
        self.d_ecid = cuda.to_device(self.ecid)
        self.update_ceid()

    def load_from_data(self):
        """
        Function to load ST data into model
        :return:
        """

        # TODO: add configuration of where to load from
        genes = list(pd.read_csv('../data_preprocessing/processed_genes.csv', header=None).squeeze())
        type_expr = pd.read_csv('../data_preprocessing/mean_expr.csv', header=None)
        type_expr.columns = genes

        ntypes = type_expr.shape[0]
        self.ntypes = ntypes

        self.expr_grid = np.loadtxt("../input_data/SpatialRef/pts.txt")
        expr_data = pd.read_csv("../input_data/SpatialRef/spatial_expr.csv", index_col=0)

        # Get list of genes that are in both datasets
        gene_set = set(genes) & set(list(expr_data))
        self.gene_set = gene_set

        # also store all ST genes
        self.st_gene_set = list(expr_data)

        print("Loaded consistent genes from SC + ST data:")
        for v in gene_set:
            print("\t", v)

        self.expr_data = expr_data.loc[:, gene_set]
        self.type_expr = type_expr.loc[:, gene_set]


        # required OT initialization for each gene
        for gene in self.gene_set:
            cur_gene_expr = np.array(expr_data[gene])
            cur_gene_expr = cur_gene_expr / np.quantile(cur_gene_expr, 0.9)
            cur_gene_expr[np.where(cur_gene_expr > 1)] = 1
            sigmas = 1.0 * np.ones(grid.shape[0], float)
            cur_data = np.concatenate((cur_gene_expr.reshape(-1, 1), sigmas.reshape(-1, 1)), axis=1)
            s.dot_initialize(self.expr_grid, cur_data, gene, self.expr_grid, device=device)

        # TODO: load cell type transition rates
        # for now just make something here
        transition = np.zeros(shape=[ntypes, ntypes])
        transition[0, 1] = 0
        transition[1, 3] = 0
        transition[3, 2] = 0
        transition[3, 4] = 0
        transition[3, 5] = 0
        transition[2, 5] = 0
        transition[4, 5] = 2.4
        transition[5, 6] = 5.4
        self.transition_matrix = transition

        # Control differential rates of cell division and death by type
        self.birth_rate_by_type = np.array([1, 1, 1, 1, 0, 0, 0])
        self.death_rate_by_type = np.array([0, 0, 0, 0, 0, 0, 1])

        ngp = self.expr_grid.shape[0]

        self.xe[:ngp, :] = self.expr_grid
        self.d_xe = cuda.to_device(self.xe)
        self.d_xe_F = cuda.to_device(self.xe)

        self.ecid[:ngp] = np.array(list(range(ngp)))
        self.ecid[ngp:] = -1
        self.d_ecid = cuda.to_device(self.ecid)
        self.update_ceid()

        # Mark the remaining elements (not covered by initial data) as inactive, and rest as active
        self.eact[:ngp] = True
        self.eact[ngp:] = False
        self.update_vact()

        # record all of the ids that are not initially in use
        for i in range(ngp, self.ne):
            self.free_ids.add(i)

        # temporarily set all cell types to non-basal
        for i in range(self.ne):
            self.ctyp[i] = 6

        # equilibriate cell positions
        #voronoi_plot(self)
        s.sem_simulation(nsteps=500, dt=dt, division=False, transition=False, dot=False)
        #voronoi_plot(self)
        # Go through each cell and decide what type it should start as
        for i in range(ngp):
            xv = self.xe[i, 0]
            yv = self.xe[i, 1]
            ymin = 3 + 3 * np.sin(2 * np.pi * xv / 26)
            if yv - ymin < 1.5:
                # basal layer - pick a random basal type
                self.ctyp[i] = np.random.choice(4)
            elif 10 - yv < 1:
                # granular layer
                self.ctyp[i] = 6
            else:
                # spinous in the middle
                if (yv - ymin)/(10-ymin) < 0.5:
                    self.ctyp[i] = 4
                else:
                    self.ctyp[i] = 5

        self.update_etyp()






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

    # profiling for debugging/optimization
    pr = cProfile.Profile()
    pr.enable()

    # each cycle consists of cell lifecycle, OT force, and SEM force
    cycles = 10
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