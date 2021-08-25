import numpy as np
from scipy.spatial import distance_matrix, Delaunay, Voronoi, voronoi_plot_2d
import seaborn as sns

#from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, MovieWriter

def simple_plot(s, pause=False, gene_color=False):
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
        # TODO: properly handle the two cases here
        if gene_color:
            a = plt.scatter(xv, yv, c=cc, vmin=0, vmax=1, cmap=sns.color_palette("vlag", as_cmap=True))
            a = plt.scatter(xv - 26, yv, c=cc, vmin=0, vmax=1,
                            cmap=sns.color_palette("vlag", as_cmap=True))
        else:
            a = plt.scatter(xv, yv, c='b')
            a = plt.scatter(xv - 26, yv, c='b')
        # a.set_aspect('equal')
        # plt.colorbar(a)
        plt.gca().set_ylim([0, 10])
        plt.gca().set_xlim([-13, 13])
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

    pointlist = []

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    def isvalid(vertex):
        x = new_vertices[vertex][0]
        y = new_vertices[vertex][1]
        ymin = 2.5 + 3*np.sin(2*np.pi*x/26)
        if y < ymin:
            return False
        else:
            return True

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
        pointlist.append(p1)
        vertices = vor.regions[region]

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]



        # if no infinite vertices, we're now done
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # otherwise, continue to dealing with infinite vertices
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

            # check for intersection with y boundary
            dist = radius

            far_point = vor.vertices[v2] + direction * dist

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return np.asarray(pointlist, dtype=np.int32), new_regions, np.asarray(new_vertices)

def voronoi_plot(s, pause=False, gene_color=False):
    # copy for periodicity
    xv = s.xe[s.eact, 0]
    yv = s.xe[s.eact, 1]
    pts1 = np.stack([xv, yv], axis=1)
    pts2 = np.stack([xv-26, yv], axis=1)
    pts = np.concatenate([pts1, pts2])



    vor = Voronoi(pts)
    pointlist, regions, vertices = voronoi_finite_polygons_2d(vor)
    plt.close('all')
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(15, 5))

    if gene_color is not False:
        if gene_color not in list(s.type_expr):
            raise ValueError
        # get information necessary for gene coloring
        # TODO: set this up with however I end up handling gene type
        cc = s.type_expr.loc[s.ctyp[s.ecid[s.eact]], gene_color]
        # old:
        # cc = s.cfeat['SPINK5'][s.ecid][s.eact]

        norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
        map = cm.ScalarMappable(norm=norm, cmap=sns.color_palette("vlag", as_cmap=True))
    else:
        # color based on cell type
        cc = s.ctyp[s.ecid[s.eact]]
        norm = mpl.colors.Normalize(vmin=0, vmax=10, clip=True)
        map = cm.ScalarMappable(norm=norm, cmap="tab10")
    # doubled for periodicity
    cc = np.concatenate([cc, cc])
    # map cell ID to the corresponding voronoi regions
    cc = cc[pointlist]

    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax[1].fill(*zip(*polygon), color=map.to_rgba(cc[i]), alpha=0.4)
        #plt.fill(*zip(*polygon))
    # plot an extra polygon to cover up the below area
    xv = np.linspace(26, -26, 200)
    yv = 2.5 + 3*np.sin(2*np.pi*xv/26)
    xv = np.concatenate([xv, np.asarray([-26, 26])])
    yv = np.concatenate([yv, np.asarray([-5, -5])])
    ax[1].fill(xv, yv, 'w')
    xv = [-26, 26, 26, -26]
    yv = [10.5, 10.5, 15, 15]
    ax[1].fill(xv, yv, 'w')
    ax[1].set_ylim([-2, 12])
    ax[1].set_xlim([-13, 13])
    ax[1].set_aspect('equal')
    type_name = ["BAS-1", "BAS-2", "BAS-3", "BAS-4", "SPN-1", "SPN-2", "GRN"]

    # plot pie chart of type distribution
    counts = np.bincount(cc, minlength=7)
    ax[0].pie(counts, labels=type_name, colors=(map.to_rgba(i) for i in range(7)))

    if pause:
        plt.pause(0.01)
    else:
        plt.show()


# Note: this assumes the points are on a uniformly spaced grid
def plot_gene_profile(pts, val, ref, gene):
    #cmap = sns.color_palette("vlag", as_cmap=True)
    cmap = "coolwarm"

    fig, ax = plt.subplots(1,2, figsize=(12, 4), dpi=300)
    cc = val
    # TODO: decide if/how to do normalization
    cc = cc/np.sum(cc)

    xv = pts[:, 0]
    yv = pts[:, 1]


    # plot the simulated gene expression data
    a = ax[1].scatter(xv, yv, c=cc, s=135, cmap=cmap)
    a = ax[1].scatter(xv - 26, yv, c=cc, s=135, cmap=cmap)
    # a.set_aspect('equal')
    # plt.colorbar(a)
    ax[1].set_ylim([-1, 11])
    ax[1].set_xlim([-13, 13])
    ax[1].set_aspect('equal')
    ax[1].set_title('simulated')

    # plot the reference data
    cc = ref
    cc = cc / np.sum(cc)
    a = ax[0].scatter(xv, yv, c=cc, s=135, cmap=cmap)
    a = ax[0].scatter(xv - 26, yv, c=cc, s=135, cmap=cmap)
    # a.set_aspect('equal')
    # plt.colorbar(a)
    ax[0].set_ylim([-1, 11])
    ax[0].set_xlim([-13, 13])
    ax[0].set_aspect('equal')
    ax[0].set_title('reference')

    fig.suptitle(gene)
    fig.savefig("../images/%s.png"%gene)
    #plt.show()
    plt.close(fig)
