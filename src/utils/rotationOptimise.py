import numpy as np
import pointVisualise as pv


def rotate_optimise(cloud):
    step = np.pi / 192

    rotY = np.array([[np.cos(step), 0, -1*np.sin(step)],
                    [0, 1, 0],
                    [np.sin(step), 0, np.cos(step)]])
    maxZval = 0
    maxZrot = 0
    for i in range(384):
        if cloud[:, 2].max() > maxZval:
            maxZval = cloud[:, 2].max()
            maxZrot = i
        cloud = rotate(cloud, rotY)
    cloud = rotate(cloud, np.array([
        [np.cos(step * maxZrot), 0, -1 * np.sin(step * maxZrot)],
        [0, 1, 0],
        [np.sin(step * maxZrot), 0, np.cos(step * maxZrot)]
    ]))

    rotX = np.array([[1, 0, 0],
                    [0, np.cos(step), -1 * np.sin(step)],
                    [0, np.sin(step), np.cos(step)]])
    maxZval = 0
    maxXrot = 0
    cloud = rotate(cloud, np.array([[1, 0, 0], [0, np.cos(step * -32), -1 * np.sin(step * -32)], [0, np.sin(step * -32), np.cos(step * -32)]]))
    for i in range(64):
        if cloud[:, 2].max() > maxZval:
            maxZval = cloud[:, 2].max()
            maxXrot = i
        cloud = rotate(cloud, rotX)
    cloud = rotate(cloud, np.array([[1, 0, 0], [0, np.cos(step * (maxXrot - 64)), -1 * np.sin(step * (maxXrot - 64))], [0, np.sin(step * (maxXrot - 64)), np.cos(step * (maxXrot - 64))]]))
    return cloud



def rotate(cloud, mat):
    tcloud = np.transpose(cloud)
    rcloud = np.matmul(mat, tcloud)
    return np.transpose(rcloud)


if __name__ == "__main__":
    import os
    os.chdir("/home/quartzshard/Documents/repos/data-driven-mask-standards")
    cloud = pv.parseObjFile("data/Matthew.obj")
    newCloud = rotate_optimise(cloud)
    pv.showPointCloud(newCloud)
