from numpy import ndarray

import pointVisualise as pv
import numpy as np
import matplotlib.pyplot as mpl

filename = r'C:\Users\keyin\Downloads\test_01.obj'
cloud = pv.parseObjFile(filename)
pv.showPointCloud(cloud)

def getLengths(cloud):
    #parameters
    nose_tol = 0.05 # nose
    face_scl = 1
    chin_bnd = 30 #
    bridg_tol = 0.5
    jaw_tol = 0.1
    trag_tol = 0.02

    # find bounding box
    max_id = np.zeros([3,1], dtype=int)
    min_id = np.zeros([3,1], dtype=int)
    max_val = np.zeros([3,1])
    min_val = np.zeros([3,1])
    for i in range(0,3):
        max_id[i] = np.argmax(cloud[:, i])
        min_id[i] = np.argmin(cloud[:, i])
        max_val[i] = cloud[max_id[i], i]
        min_val[i] = cloud[min_id[i], i]
    x_range = abs(max_val[0] - min_val[0])
    x_mid = (max_val[0] + min_val[0])/2
    nose_range = x_range*nose_tol
    x_p = x_mid + nose_range*face_scl
    x_m = x_mid - nose_range*face_scl

    # find tip of nose / max z_value
    nose = {}
    nose['idx'] = max_id[2]
    nose['coord'] = cloud[nose['idx'], :][0]
    nose['val'] = max_val[2]
    # checking if max z is a sensible value

    # find chin
    # search region just slightly above and including bottom of face
    chin_y_p = min_val[1] + chin_bnd*face_scl
    chin_condtn = np.where((cloud[:, 0] > x_m) & (cloud[:, 0] < x_p) & (cloud[:, 1] < chin_y_p))  # returns index

    chin = {}
    chin['val'] = np.amax(cloud[chin_condtn[0], 2])
    chin['idx'] = chin_condtn[0][np.argmax(cloud[chin_condtn[0], 2])]
    chin['coord'] = cloud[chin['idx'], :]

    # find bridge
    bridg_condtn = np.where((cloud[:, 0] > x_m*bridg_tol) & (cloud[:, 0] < x_p*bridg_tol) & (cloud[:, 1] > nose['coord'][1]))
    bridg_y = np.zeros([len(bridg_condtn[0]), 2])
    bridg_y[:, 0] = cloud[bridg_condtn[0], 1]
    bridg_y[:, 1] = cloud[bridg_condtn[0], 2]

    # find base of nose

    # find base jaw : find between 0.9x_max and x_max and find lowest y:
    jawp_condtn = np.where(cloud[:, 0] > max_val[0][0] - x_range*jaw_tol)
    jaw_p = {}
    jaw_p['val'] = np.amin(cloud[jawp_condtn[0], 1])
    jaw_p['idx'] = jawp_condtn[0][np.argmin(cloud[jawp_condtn[0], 1])]
    jaw_p['coord'] = cloud[jaw_p['idx'], :]

    jawm_condtn = np.where(cloud[:, 0] < min_val[0][0] + x_range*jaw_tol)
    jaw_m = {}
    jaw_m['val'] = np.amin(cloud[jawm_condtn[0], 1])
    jaw_m['idx'] = jawm_condtn[0][np.argmin(cloud[jawm_condtn[0], 1])]
    jaw_m['coord'] = cloud[jaw_m['idx'], :]

    # find tragion : same as above but for max y:
    tragp_condtn = np.where(cloud[:, 0] > max_val[0][0] - x_range*trag_tol)
    trag_p = {}
    trag_p['val'] = np.amax(cloud[tragp_condtn[0], 1])
    trag_p['idx'] = tragp_condtn[0][np.argmax(cloud[tragp_condtn[0], 1])]
    trag_p['coord'] = cloud[trag_p['idx'], :]

    jawm_condtn = np.where(cloud[:, 0] < min_val[0][0] + x_range*trag_tol)
    trag_m = {}
    trag_m['val'] = np.amax(cloud[jawm_condtn[0], 1])
    trag_m['idx'] = jawm_condtn[0][np.argmax(cloud[jawm_condtn[0], 1])]
    trag_m['coord'] = cloud[trag_m['idx'], :]

    # vector containing all points
    joined = np.zeros([3, 6])
    joined[0, 0] = trag_p['coord'][0]
    joined[0, 1] = nose['coord'][0]
    joined[0, 2] = trag_m['coord'][0]
    joined[0, 3] = jaw_m['coord'][0]
    joined[0, 4] = chin['coord'][0]
    joined[0, 5] = jaw_p['coord'][0]

    joined[1, 0] = trag_p['coord'][1]
    joined[1, 1] = nose['coord'][1]
    joined[1, 2] = trag_m['coord'][1]
    joined[1, 3] = jaw_m['coord'][1]
    joined[1, 4] = chin['coord'][1]
    joined[1, 5] = jaw_p['coord'][1]

    joined[2, 0] = trag_p['coord'][2]
    joined[2, 1] = nose['coord'][2]
    joined[2, 2] = trag_m['coord'][2]
    joined[2, 3] = jaw_m['coord'][2]
    joined[2, 4] = chin['coord'][2]
    joined[2, 5] = jaw_p['coord'][2]


    def getdistance(vec):
        len = 0
        for i in range(0,3):
            len = len + (vec[i, 0] - vec[i, 1])**2
        len = np.sqrt(len)
        return len

    # lengths
    lengths = []
    nose_trag_p = np.zeros([3, 2])
    nose_trag_p[:, 0] = nose['coord']
    nose_trag_p[:, 1] = trag_p['coord']
    nose_trag_m = np.zeros([3, 2])
    nose_trag_m[:, 0] = nose['coord']
    nose_trag_m[:, 1] = trag_m['coord']
    lengths.append((getdistance(nose_trag_m) + getdistance(nose_trag_m))/2)

    nose_chin = np.zeros([3, 2])
    nose_chin[:, 0] = nose['coord']
    nose_chin[:, 1] = chin['coord']
    lengths.append(getdistance(nose_chin))

    chin_jaw_p = np.zeros([3, 2])
    chin_jaw_p[:, 0] = chin['coord']
    chin_jaw_p[:, 1] = jaw_p['coord']
    chin_jaw_m = np.zeros([3, 2])
    chin_jaw_m[:, 0] = chin['coord']
    chin_jaw_m[:, 1] = jaw_m['coord']
    lengths.append((getdistance(chin_jaw_m) + getdistance(chin_jaw_p)) / 2)

    trag_jaw_p = np.zeros([3, 2])
    trag_jaw_p[:, 0] = trag_p['coord']
    trag_jaw_p[:, 1] = jaw_p['coord']
    trag_jaw_m = np.zeros([3, 2])
    trag_jaw_m[:, 0] = trag_m['coord']
    trag_jaw_m[:, 1] = jaw_m['coord']
    lengths.append((getdistance(trag_jaw_m) + getdistance(trag_jaw_p)) / 2)

    return joined, lengths

joined, lengths = getLengths(cloud)
mpl.plot(joined[0,:], joined[1,:], joined[2,:], '-g')
mpl.show()