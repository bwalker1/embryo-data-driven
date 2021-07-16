import numpy as np
from scipy.spatial import distance_matrix, Delaunay
import seaborn as sns

from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, MovieWriter

def simple_plot(s, pause=False, gene_color=False, periodic=False):
    # fig = plt.figure()
    plt.clf()
    if s.d == 3:
        pass
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.xe[:,0], self.xe[:,1], self.xe[:,2], c=self.etyp)
    elif s.d == 2:
        if gene_color:
            cc = s.cfeat['SPINK5'][s.ecid][s.eact]
        else:
            cc = s.ecid[s.eact]
        xv = s.xe[s.eact, 0]
        yv = s.xe[s.eact, 1]
        a = plt.scatter(xv, yv, c=cc, vmin=0, vmax=1, cmap=sns.color_palette("vlag", as_cmap=True))
        if periodic:
            a = plt.scatter(xv - 25, yv, c=cc, vmin=0, vmax=1,
                            cmap=sns.color_palette("vlag", as_cmap=True))
        # a.set_aspect('equal')
        # plt.colorbar(a)
        plt.gca().set_ylim([0, 10])
        plt.gca().set_xlim([-25, 25])
        plt.gca().set_aspect('equal')
    else:
        raise NotImplementedError
    #
    if pause:
        plt.pause(0.01)
    else:
        plt.show()

def delauney_plot_2d(s, pause=False):
    nc = np.max(s.ecid) + 1
    plt.clf()
    for i in range(nc):
        ind = np.where(s.ecid == i)[0]
        xx = s.xe[ind, 0]
        yy = s.xe[ind, 1]
        ver = Delaunay(s.xe[ind, :])
        plt.triplot(xx, yy, ver.simplices)
    if pause:
        plt.pause(0.01)
    else:
        plt.show()