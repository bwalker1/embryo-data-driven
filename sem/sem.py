import math
import numba
import torch
import pickle
from numba import cuda
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

from dot.dot import gaussian_diffusion, sinkhorn_knopp, sinkhorn_stabilized, sinkhorn_epsilon_scaling

import cProfile, pstats

# be able to adaptively run on CPU or GPU
try_gpu = True
device = torch.device('cuda:0' if (try_gpu and torch.cuda.is_available()) else 'cpu')

# TODO: describe class
class sem:
    def __init__(self, ne=100, d=3):
        self.d = d
        self.ne = ne  # number of elements
        self.xe = np.empty([self.ne, self.d], np.float32)  # coordinates of elements
        self.ecid = np.empty([self.ne], np.int32)  # cell id of elements
        self.etyp = np.empty([self.ne], np.int32)  # cell type of elements


    def initialize(self, geom='spherical', mindis=0.1, radius=10.0):
        # TODO: add general initialization control
        # Initialize elements with a spherical geometry
        if geom == 'spherical':
            # check that d=3?
            tmp_xe = np.random.uniform(-radius, radius, size=(3 * self.ne, self.d))
            tmp_d = np.linalg.norm(tmp_xe, axis=1)
            tmp_ind = np.where(tmp_d <= radius)[0]
            xe_ind = np.random.choice(tmp_ind, size=self.ne, replace=False)
            self.xe = tmp_xe[xe_ind, :]
            del tmp_xe, tmp_d, tmp_ind
        if geom == 'grid':
            if self.d != 2:
                raise ValueError
            self.xe = np.empty(shape=(self.ne, self.d))
            self.xwidth = 25
            self.xe[:, 0] = np.random.uniform(0, self.xwidth, size=self.ne)
            # initialize y coordinate based on x coordinate
            for i in range(self.ne):
                xv = self.xe[:, 0]
                ymin = 3 + 3*np.sin(2*np.pi*xv/25)
                self.xe[:, 1] = np.random.uniform(ymin, 10)
        # initalize which elements are in which cells
        self.ecid[:] = np.array(list(range(self.ne)))
        self.etyp[:] = 0
        self.d_ecid = cuda.to_device(self.ecid)
        self.d_etyp = cuda.to_device(self.etyp)
        self.d_xe = cuda.to_device(self.xe)
        self.d_xe_F = cuda.to_device(self.xe)
        self.ceid = {}
        self.update_ceid()
        self.cfeat = {}
        self.dot_P1 = {}
        self.dot_b = {}
        self.istep = 0

    def sem_simulation(self, nsteps=1000, cav=0, dt=0.01):
        if self.d != 2:
            raise NotImplementedError
        self.d_xe = cuda.to_device(self.xe)
        self.d_xe_F = cuda.to_device(self.xe)
        self.d_xe_rand = cuda.to_device(np.random.normal(0, 1, self.xe.shape))
        threads_per_block = 32
        blocks_per_grid = 64
        nc = np.max(self.ecid) + 1
        for i in range(nsteps):
            #print(i)
            self.d_xe_rand = cuda.to_device(np.random.normal(0, 1, self.xe.shape))
            cuda.synchronize()
            move_point_2[blocks_per_grid, threads_per_block](self.d_xe, self.d_xe_F, self.d_xe_rand, self.d_ecid,self.d_etyp, nc, 0.5, 1.35, dt)
            cuda.synchronize()
            self.d_xe[:, :] = self.d_xe_F[:, :]
            cuda.synchronize()
            #self.xe = move_point_2(self.xe, np.random.normal(0, 1, self.xe.shape), self.ecid, self.etyp, nc, 1.5, 3.0, 0.01)
        self.d_xe.copy_to_host(self.xe)

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
        nc = np.max(self.ecid) + 1
        for i in range(nc):
            ceid = np.where(self.ecid == i)[0]
            self.ceid[i] = ceid

    # TODO: generalize for 2d/3d geometries
    def dot_initialize(self, pts, pts_data, data_id, grids_np, dtype=torch.float32, device=torch.device("cuda")):
        # Set up the grid, respecting periodicity
        #M_np = distance_matrix(grids_np, grids_np)
        npg = grids_np.shape[0]
        M_np = np.zeros(shape=[npg, npg])
        for i in range(npg):
            for j in range(i+1, npg):
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
        P1 = gaussian_diffusion(pts_1, grids, peaks_1, sigmas_1, nus_1, x_periodic=25)
        self.dot_P1[data_id] = P1
        self.dot_b[data_id] = P1 / torch.sum(P1)
        self.dot_M = M
        self.dot_grids = grids

    def get_cell_center(self):
        nc = np.max(self.ecid) + 1
        xc = np.empty([nc, self.d], np.float32)
        for i in range(nc):
            xc[i, :] = np.mean(self.xe[self.ceid[i], :], axis=0)
        return xc

    def dot_get_gradient(self, pts_0_np, data_id, peaks=None, sigmas=None, dtype=torch.float32,
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
        P0 = gaussian_diffusion(pts_0, self.dot_grids, peaks_0, sigmas_0, nus_0)
        a = P0 / torch.sum(P0)
        D, _ = sinkhorn_knopp(a, self.dot_b[data_id], self.dot_M, 1.0)
        #print(D.cpu().item())
        D.backward()
        with torch.no_grad():
            tmp_grad = pts_0.grad
            # pts_0.grad.detach_()
            # pts_0.grad.zero_()
            # print(tmp_grad)

        return tmp_grad.cpu().numpy()

    def dot_setup(self):
        n = 128
        ctyp = np.empty([n], float)
        for i in range(n):
            tmp_eid = np.where(self.ecid == i)[0]
            if self.etyp[tmp_eid[0]] == 1:
                ctyp[i] = 1  # Te
            else:
                ctyp[i] = 0  # ICM
        self.cfeat['cell_type'] = ctyp

    def dot_simulation(self, gene):
        xc = self.get_cell_center()
        #ctyp = self.cfeat['cell_type']
        pts = []
        peaks = []
        sigmas = []
        cids = []
        nc = np.max(self.ecid)+1
        for i in range(nc):
            #if ctyp[i] == 0:
            peaks.append(self.cfeat[gene][i])
            pts.append(xc[i, :])
            cids.append(i)
            sigmas.append(1.0)
        pts = np.array(pts, float)
        peaks = np.array(peaks, float)
        sigmas = np.array(sigmas, float)
        pts_grad = self.dot_get_gradient(pts, gene, peaks=peaks, sigmas=sigmas, device=device)
        for icell in range(len(cids)):
            cid = cids[icell]
            ceid = self.ceid[cid]
            self.xe[ceid] -= pts_grad[icell, :].reshape(1, -1)
        self.d_xe = cuda.to_device(self.xe)

    def simple_plot(self, pause=False, gene_color=False, periodic=False):
        #fig = plt.figure()
        plt.clf()
        if self.d == 3:
            pass
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(self.xe[:,0], self.xe[:,1], self.xe[:,2], c=self.etyp)
        elif self.d == 2:
            if gene_color:
                cc = self.cfeat['SPINK5'][self.ecid]
            else:
                cc = self.ecid
            a = plt.scatter(self.xe[:, 0], self.xe[:, 1], c=cc, vmin=0, vmax=1, cmap=sns.color_palette("vlag", as_cmap=True))
            if periodic:
                a = plt.scatter(self.xe[:, 0]-25, self.xe[:, 1], c=cc, vmin=0, vmax=1,
                                cmap=sns.color_palette("vlag", as_cmap=True))
            #a.set_aspect('equal')
            #plt.colorbar(a)
            plt.gca().set_aspect('equal')
        else:
            raise NotImplementedError
        #
        if pause:
            plt.pause(0.01)
        else:
            plt.show()

    def polygon_plot(self):
        nc = np.max(self.ecid)+1
        ns = 50
        es = 1.5; iso=1.5
        xl = self.xe[:,0].min()-3.0; xr = self.xe[:,0].max()+3.0
        yl = self.xe[:,1].min()-3.0; yr = self.xe[:,1].max()+3.0
        x, y = np.ogrid[xl:xr:50j, yl:yr:50j]
        for i in range(nc):
            tmp_ind = np.where(self.ecid==i)[0]
            ctyp = self.etyp[tmp_ind[0]]
            tmp_pts = self.xe[tmp_ind,:]
            s = np.zeros([ns,ns,ns], float)
            for j in range(tmp_pts.shape[0]):
                pt = tmp_pts[j,:]
                s_tmp = (x-pt[0])*(x-pt[0])+(y-pt[1])*(y-pt[1])
                s += np.exp(-np.power(s_tmp/es**2,5))
            src = mlab.pipeline.scalar_field(s)
            #ctyp = self.cfeat['cell_type'][i]
            ctyp = 0
            if ctyp==1:
                mlab.pipeline.iso_surface(src, contours=[1.5, ], opacity=0.25, color=(0.5,0.5,0.5))
            elif ctyp==0:
                mlab.pipeline.iso_surface(src, contours=[3.0, ], opacity=1.0, color=(0.0,0.0,1.0))
            elif ctyp==2:
                mlab.pipeline.iso_surface(src, contours=[3.0, ], opacity=1.0, color=(1.0,0.0,0.0))
        mlab.show()

    def delauney_plot_2d(self, pause=False):
        nc = np.max(self.ecid) + 1
        plt.clf()
        for i in range(nc):
            ind = np.where(self.ecid == i)[0]
            xx = self.xe[ind, 0]
            yy = self.xe[ind, 1]
            ver = Delaunay(self.xe[ind, :])
            plt.triplot(xx, yy, ver.simplices)
        if pause:
            plt.pause(0.01)
        else:
            plt.show()

    def save_pos(self, fn):
        np.savez(fn, xe=self.xe, ecid=self.ecid, ceid=self.ceid)

    def load_pos(self, fn):
        data = np.load(fn, allow_pickle=True)
        self.xe = data['xe']
        self.ecid = data['ecid']
        self.d_ecid = cuda.to_device(self.ecid)
        self.update_ceid()


@cuda.jit
def move_point_2(x, x_F, x_rand, ecid, etyp, nc, rm_intra, rm_inter, dt):
    #x_F = x
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = x.shape[0]
    for i in range(start, ne, stride):
        for j in range(ne):
            if j == i:
                continue
            #r = math.sqrt((x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2)
            r = distance_x_periodic(x[i, 0], x[i, 1], x[j, 0], x[j, 1])
            if ecid[j] == ecid[i]:
                dV = max(d_potential_LJ(r, rm_intra, 1.5) + 0.04 * r, -50.0)
            else:
                dV = max(d_potential_LJ(r, rm_inter, 0.3), -10.0)
            x_F[i, 0] += -dt * dV * (x[i, 0] - x[j, 0])
            x_F[i, 1] += -dt * dV * (x[i, 1] - x[j, 1])
        x_F[i, 0] += dt * 0.5 * x_rand[i, 0]
        x_F[i, 1] += dt * 0.5 * x_rand[i, 1]
        d_boundary(x_F, i, dt)

    #return x_F


@cuda.jit(device=True)
def d_potential_LJ(r, rm, epsilon):
    if r == 0:
        return -10
    r2 = 1.0/(r*r)
    rs = rm/r
    rs6 = math.pow(rs,6)
    d = epsilon*r2*(-rs6*rs6+rs6)
    return d


@cuda.jit(device=True)
def d_boundary(x_F, i, dt):
    xB = 25
    yB = 10
    alpha = 25
    # box on three sides
    # if x_F[i, 0] > xB:
    #     x_F[i, 0] -= dt*alpha*(x_F[i, 0] - xB)
    # if x_F[i, 0] < 0:
    #     x_F[i, 0] -= dt*alpha*x_F[i, 0]
    # if x_F[i, 1] > yB:
    #     x_F[i, 1] -= dt*alpha*(x_F[i, 1] - yB)
    #
    # tmp = 3 + 3*math.sin(x_F[i, 0]*2*math.pi/xB)
    # if x_F[i, 1] < tmp:
    #     x_F[i, 1] -= dt*alpha*(x_F[i, 1] - tmp)

    # Periodic in x
    if x_F[i, 0] >= xB:
        x_F[i, 0] -= xB
    if x_F[i, 0] < 0:
        x_F[i, 0] += xB

    # Reflective in y
    if x_F[i, 1] > yB:
        x_F[i, 1] = yB
    tmp = 3 + 3 * math.sin(x_F[i, 0] * 2 * math.pi / xB)
    if x_F[i, 1] < tmp:
        x_F[i, 1] = tmp


@cuda.jit(device=True)
def distance_x_periodic(x1,y1,x2,y2):
    xdist = math.fabs(x1-x2)
    if xdist > 12.5:
        xdist = 25 - xdist
    return math.sqrt(xdist**2+(y1-y2)**2)


if __name__=="__main__":
    np.random.seed(5)
    s = sem(ne=208, d=2)
    s.initialize(geom='grid')

    # load spatial data
    grid = np.loadtxt("../input_data/SpatialRef/pts.txt")
    expr_data = pd.read_csv("../input_data/SpatialRef/spatial_expr_normalized.csv", index_col=0)

    spink5 = np.array(expr_data["SPINK5"])
    spink5 = spink5 / np.quantile(spink5, 0.9)
    spink5[np.where(spink5 > 1)] = 1
    sigmas = 1.0 * np.ones(grid.shape[0], float)
    data_spink5 = np.concatenate((spink5.reshape(-1, 1), sigmas.reshape(-1, 1)), axis=1)
    s.dot_initialize(grid, data_spink5, "SPINK5", grid, device=device)

    # add random gene information
    npt = len(spink5)
    nc = np.max(s.ecid)+1
    s.cfeat["SPINK5"] = np.zeros(nc, dtype=np.float64)
    xc = s.get_cell_center()
    for i in range(nc):
        # pick a random point from the spatial data and assign it to cell i
        #v = np.random.randint(0, npt)
        #s.cfeat["SPINK5"][i] = spink5[v]

        # put more stuff higher up
        s.cfeat["SPINK5"][i] = (xc[i, 1]/10)**1.7

    plt.figure(figsize=(15, 3))
    s.simple_plot(gene_color=True, pause=True, periodic=True)
    plt.figure(figsize=(15, 3))
    for i in range(200):
        print(i)
        for ii in range(10):
            #s.dot_simulation("SPINK5")
            s.sem_simulation(nsteps=100, dt=0.002)
        s.simple_plot(gene_color=True, pause=True, periodic=True)
    s.simple_plot(gene_color=True, periodic=False)