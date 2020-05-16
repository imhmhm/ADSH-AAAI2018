import numpy as np

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    top5k = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        # tindex = np.asarray(np.where(gnd == 1)) + 1.0
        tindex = np.flatnonzero(gnd == 1) + 1.0
        map_ = np.mean(count / (tindex))
        top5k_ = np.sum(gnd[0:5000]) / 5000.0
        # print(map_)
        map = map + map_
        top5k = top5k + top5k_

    map = map / num_query
    top5k = top5k / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map, top5k

def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    top5k = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        # tindex = np.asarray(np.where(tgnd == 1)) + 1.0 # np.where(condition) return indices
        tindex = np.flatnonzero(tgnd == 1) + 1.0
        topkmap_ = np.mean(count / (tindex))

        top5k_ = tsum / 5000.0
        # print(topkmap_)
        topkmap = topkmap + topkmap_
        top5k = top5k + top5k_

    topkmap = topkmap / num_query
    top5k = top5k / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap, top5k

if __name__=='__main__':
    pass
