import math
import torch
import pickle
from numba import cuda
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

from sem.dot.dot import gaussian_diffusion, sinkhorn_knopp

# for testing/optimizing
import cProfile

# be able to adaptively run on CPU or GPU
try_gpu = True
device = torch.device('cuda:0' if (try_gpu and torch.cuda.is_available()) else 'cpu')


class sem:
    def __init__(self, ne=100, d=3):
        self.d = d
        self.ne = ne                                        # number of elements
        self.xe = np.empty([self.ne, self.d], np.float32)   # coordinates of elements
        self.ecid = np.empty([self.ne], np.int32)           # cell id of elements
        self.etyp = np.empty([self.ne], np.int32)           # cell type of elements

    def initialize(self, geom='spherical', mindis=0.1, radius=10.0):
        # TODO: add general initialization control
        # Initialize elements with a spherical geometry
        if geom == 'spherical':
            # check that d=3?
            tmp_xe = np.random.uniform(-radius, radius, size=(3*self.ne, self.d))
            tmp_d = np.linalg.norm(tmp_xe, axis=1)
            tmp_ind = np.where(tmp_d <= radius)[0]
            xe_ind = np.random.choice(tmp_ind, size=self.ne, replace=False)
            self.xe = tmp_xe[xe_ind, :]
            del tmp_xe, tmp_d, tmp_ind
        self.ecid[:] = 0
        self.etyp[:] = 0
        self.d_ecid = cuda.to_device(self.ecid)
        self.d_etyp = cuda.to_device(self.etyp)
        self.d_xe = cuda.to_device(self.xe)
        self.d_xe_F = cuda.to_device(self.xe)
        self.ceid = {}
        self.ceid[0] = np.where(self.ecid == 0)[0]
        self.cfeat = {}
        self.dot_P1 = {}
        self.dot_b = {}
        self.istep = 0

    def cell_division(self, cid, type):
        """Divide cell of cid into two.
        """
        # Divide the a cell without adding elements
        if type == 1:
            # split the cell along a random axis
            vec = np.random.rand(3)
            eid = np.where(self.ecid == cid)[0]
            nc = np.max(self.ecid)+1
            dis = self.xe[eid, :].dot(vec.reshape(-1, 1))
            mdis = np.median(dis)
            tmp_eid = np.where(dis <= mdis)[0]
            self.ecid[eid[tmp_eid]] = nc
        self.d_ecid = cuda.to_device(self.ecid)
        self.update_ceid()
        cuda.synchronize()

    def update_ceid(self):
        nc = np.max(self.ecid)+1
        for i in range(nc):
            ceid = np.where(self.ecid == i)[0]
            self.ceid[i] = ceid

    def decide_ctyp(self, n_inner_cell):
        nc = np.max(self.ecid)+1
        # Compute the centroid location of each cell
        xc = np.empty([nc, self.d], np.float32)
        for i in range(nc):
            ceid = np.where(self.ecid == i)[0]
            xc[i, :] = np.mean(self.xe[ceid, :], axis=0)
            self.ceid[i] = ceid
        xd = np.linalg.norm(xc,axis=1)
        # print(xd, xc)
        sorted_cid = np.argsort(-xd)
        # assign cells closer to the center to be inner cells
        for i in sorted_cid[:nc-n_inner_cell]:
            ceid = np.where(self.ecid==i)[0]
            self.etyp[ceid] = 1
        self.d_etyp = cuda.to_device(self.etyp)
        cuda.synchronize()

    def sem_simulation(self, nsteps=1000, cav=0):
        self.d_xe = cuda.to_device(self.xe)
        self.d_xe_F = cuda.to_device(self.xe)
        self.d_xe_rand = cuda.to_device(np.random.normal(0, 1, self.xe.shape))
        threads_per_block = 128
        blocks_per_grid = 32
        nc = np.max(self.ecid)+1
        for i in range(nsteps):
            # if i%10==0: print(i)
            cuda.synchronize()
            move_point[blocks_per_grid, threads_per_block](self.d_xe, self.d_xe_F, self.d_xe_rand, self.d_ecid, self.d_etyp, nc, 1.5, 3.0, 0.01, cav)
            cuda.synchronize()
            self.d_xe[:, :] = self.d_xe_F[:, :]
            cuda.synchronize()
        self.d_xe.copy_to_host(self.xe)

    def dot_initialize(self, pts, pts_data, data_id, dtype=torch.float32, device=torch.device("cuda")):
        # Set up the grid
        ngrid = 15
        xl = yl = zl = -16.0; xr = yr = zr = 16.0
        x = np.linspace(xl, xr, ngrid)
        y = np.linspace(yl, yr, ngrid)
        z = np.linspace(zl, zr, ngrid)
        xv, yv, zv = np.meshgrid(x, y, z)
        grids_np = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1),
            zv.reshape(-1,1)), axis=1)
        M_np = distance_matrix(grids_np, grids_np)

        # Set up reference data
        npt_1 = pts.shape[0]
        nus_1_np = np.ones(npt_1)
        pts_1 = torch.tensor(pts, dtype=dtype, device=device)
        sigmas_1 = torch.tensor(pts_data[:,1], dtype=dtype, device=device)
        peaks_1 = torch.tensor(pts_data[:,0], dtype=dtype, device=device)
        nus_1 = torch.tensor(nus_1_np, dtype=dtype, device=device)

        grids = torch.tensor(grids_np, dtype=dtype, device=device)
        M = torch.tensor(M_np, dtype=dtype, device=device)
        P1 = gaussian_diffusion(pts_1, grids, peaks_1, sigmas_1, nus_1)
        self.dot_P1[data_id] = P1
        self.dot_b[data_id] = P1/torch.sum(P1)
        self.dot_M = M
        self.dot_grids = grids

    
    def get_cell_center(self):
        nc = np.max(self.ecid) + 1
        xc = np.empty([nc, 3], np.float32)
        for i in range(nc):
            xc[i,:] = np.mean(self.xe[self.ceid[i], :], axis=0)
        return xc

    def dot_get_gradient(self, pts_0_np, data_id, peaks=None, sigmas=None, dtype=torch.float32, device=torch.device("cuda")):
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
        a = P0/torch.sum(P0)
        D,_ = sinkhorn_knopp(a, self.dot_b[data_id], self.dot_M, 1.0)
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
            tmp_eid = np.where(self.ecid==i)[0]
            if self.etyp[tmp_eid[0]] == 1:
                ctyp[i] = 1 # Te
            else:
                ctyp[i] = 0 # ICM
        self.cfeat['cell_type'] = ctyp

    def dot_simulation(self, gene):
        xc = self.get_cell_center()
        ctyp = self.cfeat['cell_type']
        pts = []
        peaks = []
        sigmas = []
        cids = []
        for i in range(128):
            if ctyp[i] == 0:
                peaks.append(self.cfeat[gene][i])
                pts.append(xc[i,:])
                cids.append(i)
                sigmas.append(1.5)
        pts = np.array( pts, float )
        peaks = np.array( peaks, float )
        sigmas = np.array( sigmas, float )
        pts_grad = self.dot_get_gradient(pts, gene, peaks=peaks, sigmas=sigmas, device=device)
        for icell in range(len(cids)):
            cid = cids[icell]
            ceid = self.ceid[cid]
            self.xe[ceid] -= pts_grad[icell,:].reshape(1,-1)
        self.d_xe = cuda.to_device(self.xe)

    def simple_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.xe[:,0], self.xe[:,1], self.xe[:,2], c=self.etyp)
        plt.show()

    def polygon_plot(self):
        nc = np.max(self.ecid)+1
        ns = 50
        es = 1.5; iso=1.5
        xl = self.xe[:,0].min()-3.0; xr = self.xe[:,0].max()+3.0
        yl = self.xe[:,1].min()-3.0; yr = self.xe[:,1].max()+3.0
        zl = self.xe[:,2].min()-3.0; zr = self.xe[:,2].max()+3.0
        x, y, z = np.ogrid[xl:xr:50j, yl:yr:50j, zl:zr:50j]
        for i in range(nc):
            tmp_ind = np.where(self.ecid==i)[0]
            ctyp = self.etyp[tmp_ind[0]]
            tmp_pts = self.xe[tmp_ind,:]
            s = np.zeros([ns,ns,ns], float)
            for j in range(tmp_pts.shape[0]):
                pt = tmp_pts[j,:]
                s_tmp = (x-pt[0])*(x-pt[0])+(y-pt[1])*(y-pt[1])+(z-pt[2])*(z-pt[2])
                s += np.exp(-np.power(s_tmp/es**2,5))
            src = mlab.pipeline.scalar_field(s)
            ctyp = self.cfeat['cell_type'][i]
            if ctyp==1:
                mlab.pipeline.iso_surface(src, contours=[1.5, ], opacity=0.25, color=(0.5,0.5,0.5))
            elif ctyp==0:
                mlab.pipeline.iso_surface(src, contours=[3.0, ], opacity=1.0, color=(0.0,0.0,1.0))
            elif ctyp==2:
                mlab.pipeline.iso_surface(src, contours=[3.0, ], opacity=1.0, color=(1.0,0.0,0.0))
        mlab.show()

    def polygon_plot_gene(self, gene):
        nc = np.max(self.ecid)+1
        ns = 50
        es = 1.5; iso=1.5
        xl = self.xe[:,0].min()-3.0; xr = self.xe[:,0].max()+3.0
        yl = self.xe[:,1].min()-3.0; yr = self.xe[:,1].max()+3.0
        zl = self.xe[:,2].min()-3.0; zr = self.xe[:,2].max()+3.0
        x, y, z = np.ogrid[xl:xr:50j, yl:yr:50j, zl:zr:50j]
        norm = mpl.colors.Normalize(vmin=0.2, vmax=0.8)
        scalarMap = cm.ScalarMappable(norm=norm, cmap="coolwarm")
        for i in range(nc):
            tmp_ind = np.where(self.ecid==i)[0]
            ctyp = self.etyp[tmp_ind[0]]
            tmp_pts = self.xe[tmp_ind,:]
            s = np.zeros([ns,ns,ns], float)
            for j in range(tmp_pts.shape[0]):
                pt = tmp_pts[j,:]
                s_tmp = (x-pt[0])*(x-pt[0])+(y-pt[1])*(y-pt[1])+(z-pt[2])*(z-pt[2])
                s += np.exp(-np.power(s_tmp/es**2,5))
            src = mlab.pipeline.scalar_field(s)
            ctyp = self.cfeat['cell_type'][i]
            if ctyp==1:
                mlab.pipeline.iso_surface(src, contours=[1.5, ], opacity=0.25, color=(0.5,0.5,0.5))
            else:
                color = scalarMap.to_rgba(self.cfeat[gene][i])
                mlab.pipeline.iso_surface(src, contours=[3.0, ], opacity=1.0, color=color[:3])
        mlab.show()


@cuda.jit
def move_point(x, x_F, x_rand, ecid, etyp, nc, rm_intra, rm_inter, dt, cav):
    xB_out = 15.0; xB_in = 12.0
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = x.shape[0]
    for i in range(start, ne, stride):
        ictyp = etyp[i]
        ro = math.sqrt( x[i,0]**2 + x[i,1]**2 + x[i,2]**2 )
        if cav==1:
            tmp_1, tmp_2, tmp_3 = d_potential_cav(0.0,0.0,1.0,x[i,0],x[i,1],x[i,2],ictyp,xB_out)
            x_F[i,0] -= 5.0*dt*tmp_1; x_F[i,1] -= 5.0*dt*tmp_2; x_F[i,2] -= 5.0*dt*tmp_3
        for j in range(ne):
            if j == i: continue
            r = math.sqrt( (x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2 + (x[i,2]-x[j,2])**2 )
            if ecid[j] == ecid[i]:
                dV = max( d_potential_LJ(r, rm_intra, 1.5) + 0.01*r, -10.0 )
            else:
                dV = max( d_potential_LJ(r, rm_inter, 0.3), -10.0 )
            x_F[i,0] += -dt * dV * (x[i,0] - x[j,0])
            x_F[i,1] += -dt * dV * (x[i,1] - x[j,1])
            x_F[i,2] += -dt * dV * (x[i,2] - x[j,2])
        x_F[i,0] += dt * 0.5 * x_rand[i,0]
        x_F[i,1] += dt * 0.5 * x_rand[i,1]
        x_F[i,2] += dt * 0.5 * x_rand[i,2]
        if nc < 32:
            x_F[i,0] -= 0.001 * x[i,0]
            x_F[i,1] -= 0.001 * x[i,1]
            x_F[i,2] -= 0.001 * x[i,2]
        if ( ro>xB_out and ictyp==1 ):
            x_F[i,0] = x_F[i,0]*xB_out/ro
            x_F[i,1] = x_F[i,1]*xB_out/ro
            x_F[i,2] = x_F[i,2]*xB_out/ro
        if ( ro>xB_in and ictyp==0 ):
            x_F[i,0] = x_F[i,0]*xB_in/ro
            x_F[i,1] = x_F[i,1]*xB_in/ro
            x_F[i,2] = x_F[i,2]*xB_in/ro
    
@cuda.jit(device=True)
def d_potential_LJ(r, rm, epsilon):
    r2 = 1.0/(r*r)
    rs = rm/r
    rs6 = math.pow(rs,6)
    d = epsilon*r2*(-rs6*rs6+rs6)
    return d

@cuda.jit(device=True)
def d_potential_cav(xcav, ycav, zcav, x, y, z, ctyp, R):
    # Outer cell
    if ctyp==1:
        r = math.sqrt( x**2 + y**2 + z**2 )
        xR = x*R/r; yR = y*R/r; zR = z*R/r
        te_repulsion = 0.0
        if r < R: te_repulsion = 40.0/(R-r)
        te_repulsion = max(-50.0, te_repulsion)
        te_repulsion = min(50.0, te_repulsion)
        x_change = (x-xR)*te_repulsion
        y_change = (y-yR)*te_repulsion
        z_change = (z-zR)*te_repulsion
    # Inner cell
    elif ctyp == 0:
        # Auxiliary point repulsion center for cavity
        xc_rep = R*xcav; yc_rep = R*ycav; zc_rep = R*zcav
        rr = distance(x,y,z,xc_rep,yc_rep,zc_rep)
        mm = 1.0/rr
        xc_rep_1 = 0.5*xc_rep + 0.86*zc_rep
        xc_rep_2 = 0.5*xc_rep - 0.86*zc_rep
        yc_rep_1 = 0.5*yc_rep + 0.85*zc_rep
        yc_rep_2 = 0.5*yc_rep - 0.86*zc_rep
        zc_rep_1 = 0.5*zc_rep - 0.86*xc_rep
        zc_rep_2 = 0.5*zc_rep + 0.86*xc_rep
        zc_rep_3 = 0.5*zc_rep - 0.86*yc_rep
        zc_rep_4 = 0.5*zc_rep + 0.86*yc_rep
        rr = distance(x,y,z,xc_rep_1,yc_rep,zc_rep_1)
        mm1 = 1.0/rr
        rr = distance(x,y,z,xc_rep_2,yc_rep,zc_rep_2)
        mm2 = 1.0/rr
        rr = distance(x,y,z,xc_rep,yc_rep_1,zc_rep_3)
        mm3 = 1.0/rr
        rr = distance(x,y,z,xc_rep,yc_rep_2,zc_rep_4)
        mm4 = 1.0/rr
        ro = distance(x,y,z,0.0,0.0,0.0)
        mm  = min(50.0, mm )
        mm1 = min(50.0, mm1)
        mm2 = min(50.0, mm2)
        mm3 = min(50.0, mm3)
        mm4 = min(50.0, mm4)
        x_change = (x-xc_rep)*(mm+mm3+mm4) \
            + (x-xc_rep_1)*mm1 + (x-xc_rep_2)*mm2
        y_change = (y-yc_rep)*(mm+mm1+mm2) \
            + (y-yc_rep_1)*mm3 + (y-yc_rep_2)*mm4
        z_change = (z-zc_rep)*mm + (z-zc_rep_1)*mm1 \
            + (z-zc_rep_2)*mm2 + (z-zc_rep_3)*mm3 \
            + (z-zc_rep_4)*mm4
    return x_change, y_change, z_change


@cuda.jit(device=True)
def distance(x1,y1,z1,x2,y2,z2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)


def polygon_plot_gene(pts, data):
    nc = pts.shape[0]
    ns = 50
    es = 1.5; iso=0.5
    xl = -15-3.0; xr = 15+3.0
    yl = -15-3.0; yr = 15+3.0
    zl = -15-3.0; zr = 15+3.0
    x, y, z = np.ogrid[xl:xr:50j, yl:yr:50j, zl:zr:50j]
    norm = mpl.colors.Normalize(vmin=0.2, vmax=0.8)
    scalarMap = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    for i in range(nc):
        tmp_pts = pts[i,:].reshape(1,-1)
        s = np.zeros([ns,ns,ns], float)
        for j in range(tmp_pts.shape[0]):
            pt = tmp_pts[j,:]
            s_tmp = (x-pt[0])*(x-pt[0])+(y-pt[1])*(y-pt[1])+(z-pt[2])*(z-pt[2])
            s += np.exp(-np.power(s_tmp/es**2,5))
        src = mlab.pipeline.scalar_field(s)
        color = scalarMap.to_rgba(data[i])
        mlab.pipeline.iso_surface(src, contours=[iso, ], opacity=1.0, color=color[:3])
    mlab.show()


def ddsem_simulation(Seq32C, Seq64C, Seq128C, Spa128C, random_seed):
    np.random.seed(random_seed)
    spa_datadir = "./input_data"
    seq_datadir = "./input_data"
    # Initialize a zygote
    embryo = sem(ne=1280)
    embryo.initialize()
    # Read in the spatial reference data
    df_spa128c = pd.read_csv(spa_datadir+"/"+Spa128C+".txt", sep="\t")
    pts_spa128c = np.array( df_spa128c[["x","y","z"]] )
    pts_spa128c = pts_spa128c
    Nanog_spa128c = np.array( df_spa128c["Nanog_avg"] )
    Gata6_spa128c = np.array( df_spa128c["Gata6_avg"] )
    Nanog_spa128c = Nanog_spa128c/np.quantile(Nanog_spa128c, 0.9); Nanog_spa128c[np.where(Nanog_spa128c>1)] = 1
    Gata6_spa128c = Gata6_spa128c/np.quantile(Gata6_spa128c, 0.9); Gata6_spa128c[np.where(Gata6_spa128c>1)] = 1
    sigmas_spa128c = 1.5 * np.ones(pts_spa128c.shape[0], float)
    Data_Nanog = np.concatenate((Nanog_spa128c.reshape(-1, 1), sigmas_spa128c.reshape(-1, 1)), axis=1)
    Data_Gata6 = np.concatenate((Gata6_spa128c.reshape(-1, 1), sigmas_spa128c.reshape(-1, 1)), axis=1)
    embryo.dot_initialize(pts_spa128c, Data_Nanog, "Nanog", device=device)
    embryo.dot_initialize(pts_spa128c, Data_Gata6, "Gata6", device=device)
    # polygon_plot_gene(pts_spa128c, Data_Nanog[:,0])
    # Read in the sequencing data
    infile = open(seq_datadir+"/devmappath_"+Seq64C+"-"+Seq128C+".pkl", "rb")
    seq_path = pickle.load(infile)
    common_genes = list(np.loadtxt(seq_datadir+"/common_genes.txt", dtype=str))
    # Simulation
    # 1c-128c without data
    nc = 1
    while nc < 128:
        print("nc: ", nc)
        for j in range(nc):
            embryo.cell_division(j,1)
        embryo.sem_simulation(nsteps=1000, cav=0)
        nc = np.max(embryo.ecid)+1
    # Assign inner cells at 128c
    embryo.decide_ctyp(len(seq_path))
    embryo.sem_simulation(nsteps=1000, cav=1)
    # embryo.simple_plot()
    # OT driven simulation
    nicm_seq = len(seq_path)
    embryo.dot_setup()
    ind = np.arange(nicm_seq)
    ind = np.random.permutation(ind)
    seq_Nanog = np.empty(nicm_seq)
    nintervals = -1
    x = np.arange(nintervals+1)/float(nintervals)
    seq_Nanog = np.empty([nicm_seq, len(x)], float)
    for i in range(nicm_seq):
        tmp_Nanog = seq_path[i][common_genes.index("Nanog"),:]
        seq_Nanog[i,:] = np.interp(x, np.arange(len(tmp_Nanog))/(len(tmp_Nanog)-1), tmp_Nanog)
    seq_Nanog = seq_Nanog[ind,:]

    embryo.cfeat['Nanog'] = np.zeros(128)
    pts_traj = np.empty([nintervals+1, nicm_seq, 3], float)
    print("Max intervals: ", nintervals)
    for i in range(nintervals+1):
        print("Interval: ", i)
        cnt = 0
        for ic in range(128):
            if embryo.cfeat['cell_type'][ic] == 0:
                embryo.cfeat['Nanog'][ic] = seq_Nanog[cnt,i]
                cnt += 1
       
        for j in range(1000):
            embryo.dot_simulation("Nanog")
            embryo.sem_simulation(cav=0, nsteps=10)

        embryo.sem_simulation(cav=1, nsteps=20)
        
        xc = embryo.get_cell_center()
        ctyp = embryo.cfeat['cell_type']
        cnt = 0
        for k in range(128):
            if ctyp[k] == 0:
                pts_traj[i,ind[cnt],:] = xc[k,:]
                cnt += 1
    np.save("simulation_results/pts_traj_"+Seq64C+"_"+Seq128C+"_"+str(random_seed)+".npy", pts_traj)


def main():
    for i in range(1):
        print("Running with random seed ", str(i))
        ddsem_simulation(None, "Q_C64E1", "G_E4.5Em4", "020614_R3_C2", i)



if __name__ == "__main__":
    cProfile.run('main()')
