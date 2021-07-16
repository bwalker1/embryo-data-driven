import numpy as np
from scipy.spatial import distance_matrix, Delaunay, Voronoi, voronoi_plot_2d
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
            a = plt.scatter(xv - 26, yv, c=cc, vmin=0, vmax=1,
                            cmap=sns.color_palette("vlag", as_cmap=True))
        # a.set_aspect('equal')
        # plt.colorbar(a)
        plt.gca().set_ylim([0, 10])
        plt.gca().set_xlim([-26, 26])
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



# Adapted from https://gist.github.com/pv/8036995
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def voronoi_plot(s, pause=False):
    # copy for periodicity
    xv = s.xe[s.eact, 0]
    yv = s.xe[s.eact, 1]
    pts1 = np.stack([xv, yv], axis=1)
    pts2 = np.stack([xv-26, yv], axis=1)
    pts = np.concatenate([pts1, pts2])

    vor = Voronoi(pts1)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # post-process to extract 1 full period and draw boundary

    plt.clf()
    ax = plt.gca()
    voronoi_plot_2d(vor, ax=ax)
    if pause:
        plt.pause(0.01)
    else:
        plt.show()
