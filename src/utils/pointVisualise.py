import numpy as np
import matplotlib.pyplot as mpl
import random


def showPointCloud(cloud, plotColour=None, plotColourForMyFunction=None, resolution=10):
    """
    Show the point cloud (np array) with matplotlob 3D Scatter, where the array is a (3,n) array of [x,y,z] points
    """
    fig = mpl.figure()
    ax = fig.add_subplot(projection="3d")
    strippedCloud = cloud[::resolution, :]
    if plotColour:
        print("ahsda")
        ax.scatter(strippedCloud[:, 0], strippedCloud[:, 1], strippedCloud[:, 2], c=strippedCloud[:, 3:]/255)
    elif plotColourForMyFunction:
        r_offset = random.randint(0, 255)
        g_offset = random.randint(0, 255)
        b_offset = random.randint(0, 255)
        strippedCloud[:, 6] += 1
        max_cluster_name = strippedCloud[:, 6].max()
        print(max_cluster_name)
        strippedCloud[:, 6] = strippedCloud[:, 6] * 255 / (max_cluster_name+1)
        colour_array = np.array([(strippedCloud[:, 6] + r_offset).astype(int) % 255, (strippedCloud[:, 6] + g_offset).astype(int) % 255, (strippedCloud[:, 6] + b_offset).astype(int) % 255]).T
        print(colour_array.shape)
        ax.scatter(strippedCloud[:, 0], strippedCloud[:, 1], strippedCloud[:, 2], c=colour_array/255)#(strippedCloud[:, 6]).astype(int) + r_offset, (strippedCloud[:, 6]).astype(int) + g_offset, (strippedCloud[:, 6]).astype(int) + b_offset))
    else:
        ax.scatter(strippedCloud[:, 0], strippedCloud[:, 1], strippedCloud[:, 2])
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

def parse_2d():
    import pandas as pd

    facemap = pd.read_csv("data/from2d/face_points_matthew.csv", usecols=[1, 2, 3]).to_numpy()
    facemap[:, 0] -= facemap[:, 0].mean()
    facemap[:, 1] -= facemap[:, 1].mean()
    facemap[:, 2] -= facemap[:, 2].mean()

    showPointCloud(facemap, resolution=1)
    pass


if __name__ == "__main__":
    # For pycharm, give it the path to the repo here
    # import os
    # os.chdir("<path>/data-driven-mask-standards")

    # cloud = parseObjFile("data/scans/FaMoS_180424_03335_TA/natural_head_rotation.000001.obj")
    cloud = parsePtsFile("data/Matthew.pts")
    showPointCloud(cloud, plotColour=True, resolution=10)
