import numpy as np


def compute_iou(bbox1, bbox2):
    """
    Compute the intersection over union (IoU) of two bounding boxes

    Args:
        bbox1: bounding box 1, numpy array of shape (4,) with format (x1, y1, x2, y2)
        bbox2: bounding box 2, numpy array of shape (4,) with format (x1, y1, x2, y2)

    Returns:
        iou: IoU of bbox1 and bbox2
    """
    # Convert xywh to x1y1x2y2
    bbox1 = np.array([bbox1[0], bbox1[1], bbox1[0] +
                     bbox1[2], bbox1[1]+bbox1[3]])
    bbox2 = np.array([bbox2[0], bbox2[1], bbox2[0] +
                     bbox2[2], bbox2[1]+bbox2[3]])
    # Compute the intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection_area = max(0, x2-x1+1)*max(0, y2-y1+1)

    # Compute the union area
    bbox1_area = (bbox1[2]-bbox1[0]+1)*(bbox1[3]-bbox1[1]+1)
    bbox2_area = (bbox2[2]-bbox2[0]+1)*(bbox2[3]-bbox2[1]+1)
    union_area = bbox1_area+bbox2_area-intersection_area

    # Compute the IoU
    iou = intersection_area/(union_area + 1e-8)

    return iou


def compute_distr_and_avg(data, bins=5):
    hist, bin_edges = np.histogram(data, bins=bins)
    avg = np.mean(data)
    return hist / np.sum(hist), bin_edges, avg
