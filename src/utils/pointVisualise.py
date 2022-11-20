import numpy as np
import matplotlib.pyplot as mpl


def showPointCloud(cloud):
    """
    Show the point cloud (np array) with matplotlob 3D Scatter, where the array is a (3,n) array of [x,y,z] points
    """
    fig = mpl.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(cloud[::10, 0], cloud[::10, 1], cloud[::10, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    mpl.show()


def parseObjFile(filename):
    """
    Loads a .obj from <filename> to a (3,n) numpy array.
    :rtype: object
    """
    with open(filename) as file:
        toarr = []
        for line in file.readlines():
            if line[0] != "v" or line[1] != " ":
                continue
            line = line[2:-2]
            toarr.append(line.split(" "))
        for i in toarr:
            for j in range(len(i)):
                try:
                    i[j] = float(i[j])
                except:
                    i[j] = float(0)
        cloud = np.array(toarr)
        cloud[:, 0] -= cloud[:, 0].mean()
        cloud[:, 1] -= cloud[:, 1].mean()
        cloud[:, 2] -= cloud[:, 2].mean()
        return cloud


def parsePtsFile(filename):
    """
    Loads a .pts from <filename> to a (6,n) numpy array.
    """
    with open(filename) as file:
        rows = [rows.strip() for rows in file]
    rows.pop(0)
    toarr = []
    for row in rows:
        toarr.append(row.split()[0:3] + row.split()[4:7])
    cloud = np.array(toarr, dtype="float")
    cloud[:, 0] -= cloud[:, 0].mean()
    cloud[:, 1] -= cloud[:, 1].mean()
    cloud[:, 2] -= cloud[:, 2].mean()
    return cloud


if __name__ == "__main__":
    # For pycharm, give it the path to the repo here
    # import os
    # os.chdir("<path>/data-driven-mask-standards")

    # cloud = parseObjFile("data/scans/FaMoS_180424_03335_TA/natural_head_rotation.000001.obj")
    # cloud = parsePtsFile("data/Matthew.pts")
    cloud = parseObjFile(r'C:\Users\keyin\Downloads\test_01.obj')
    showPointCloud(cloud)