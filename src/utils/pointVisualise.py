import numpy as np
import matplotlib.pyplot as mpl

def showPointCloud(cloud):
    """
    Show the point cloud (np array) with matplotlob 3D Scatter, where the array is a (3,n) array of [x,y,z] points
    """
    fig = mpl.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(cloud[::10,0],cloud[::10,1], cloud[::10,2])
    mpl.show()

def parseObjFile(filename):
    """
    Loads a .obj from <filename> to a (3,n) numpy array.
    """
    with open(filename) as file:
        toarr = []
        for line in file.readlines():
            if line[0] != "v":
                continue
            line = line[2:-2]
            toarr.append(line.split(" "))
        for i in toarr:
            for j in range(len(i)):
                try:
                    i[j] = float(i[j])
                except:
                    i[j] = float(0)
        cloud = np.array(toarr, dtype="float")
        return cloud

if __name__ == "__main__":
    cloud = parseObjFile("data/scans/FaMoS_180424_03335_TA/natural_head_rotation.000001.obj")
    showPointCloud(cloud)
   