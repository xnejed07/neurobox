import numpy as np


def filter_short_segments(detections, w: int) -> list:
    return [k for k in detections if (k[1] - k[0]) >= w]


def concat_close_segments(detections, w):
    concated = []
    for k in detections:
        if len(concated) == 0:
            concated.append(k)
        else:
            if concated[-1][1] + w >= k[0]:
                concated[-1] = (concated[-1][0], k[1])
            else:
                concated.append(k)
    return concated

def threshold_and_find_segments(x,threshold):
    x[x>=threshold] = 1
    x[x!=1] = 0
    x[0] = 0
    x[-1] = 0
    out = []
    start = None
    for i in range(x.shape[0]-1):
        if (x[i] == 0) and (x[i+1] == 1):
            start = i
        if (x[i] == 1) and (x[i + 1] == 0):
            out.append((start,i))
            start = None
    return out