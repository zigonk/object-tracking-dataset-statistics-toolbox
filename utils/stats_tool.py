from type.TrackingType import TrackingData
from type.StatsNameEnum import StatsName
from typing import List

import numpy as np

from utils.utils import compute_distr_and_avg, compute_iou

# -------------------------------------------------- VIDEO INFORMATION STATISTICS --------------------------------------------------#


def count_obj_per_video(gt_tracking: List[TrackingData], bins=5):
    """
    Count the number of objects per video and return a histogram of the counts and average number of objects per video

    Args:
        gt_tracking: list of tracking ground truth data
        bins: number of bins for the histogram or bin edges

    Returns:
        hist: histogram of the number of objects per video
        bin_edges: bin edges for the histogram
        avg_num_objs: average number of objects per video
    """
    num_objs = []
    for gt in gt_tracking:
        obj_ids = np.unique(gt.data[:, 1])
        num_objs.append(len(obj_ids))

    return compute_distr_and_avg(num_objs, bins=bins)


def count_obj_per_frame(gt_tracking: List[TrackingData], bins=5):
    """
    Count the number of objects per frame and return a histogram of the counts and average number of objects per frame

    Args:
        gt_tracking: list of tracking ground truth data
        bins: number of bins for the histogram or bin edges

    Returns:
        hist: histogram of the number of objects per frame
        bin_edges: bin edges for the histogram
        avg_num_objs: average number of objects per frame
    """
    num_objs = []
    for gt in gt_tracking:
        frame_ids = np.unique(gt.data[:, 0])
        for frame_id in frame_ids:
            num_objs.append(len(gt.data[gt.data[:, 0] == frame_id]))

    return compute_distr_and_avg(num_objs, bins=bins)


def compute_video_length(gt_tracking: List[TrackingData], bins=5):
    """
    Compute the length of the videos and return a histogram of the lengths and average length

    Args:
        gt_tracking: list of tracking ground truth data
        bins: number of bins for the histogram or bin edges

    Returns:
        hist: histogram of the video lengths
        bin_edges: bin edges for the histogram
        avg_video_length: average video length
    """
    video_lengths = []
    for gt in gt_tracking:
        video_lengths.append(len(np.unique(gt.data[:, 0])))

    return compute_distr_and_avg(video_lengths, bins=bins)

# -------------------------------------------------- TRACKING CHALLENGE STATISTICS --------------------------------------------------#


def compute_track_gap_length(gt_tracking: List[TrackingData], bins=5):
    """
    Compute the length of the gaps in the tracks and return a histogram of the gap lengths and average gap length

    Args:
        gt_tracking: list of tracking ground truth data
        bins: number of bins for the histogram or bin edges

    Returns:
        hist: histogram of the gap lengths
        bin_edges: bin edges for the histogram
        avg_gap_length: average gap length
    """
    gap_lengths = []
    for gt in gt_tracking:
        obj_ids = np.unique(gt.data[:, 1])
        for obj_id in obj_ids:
            obj_data = gt.data[gt.data[:, 1] == obj_id]
            frame_ids = obj_data[:, 0]
            for i in range(len(frame_ids)-1):
                gap = frame_ids[i+1]-frame_ids[i]-1
                if gap > 0:
                    gap_lengths.append(frame_ids[i+1]-frame_ids[i]-1)

    return compute_distr_and_avg(gap_lengths, bins=bins)


def compute_iou_ratio_objects_intra_frame(gt_tracking: List[TrackingData], bins=5):
    """
    Compute the ratio of the intersection over union (IoU) of the bounding boxes of objects in the same frame and return a histogram of the ratios and average ratio

    Args:
        gt_tracking: list of tracking ground truth data
        bins: number of bins for the histogram or bin edges

    Returns:
        hist: histogram of the IoU ratios
        bin_edges: bin edges for the histogram
        avg_iou_ratio: average IoU ratio
    """
    iou_ratios = []
    for gt in gt_tracking:
        frame_ids = np.unique(gt.data[:, 0])
        for frame_id in frame_ids:
            frame_bboxes = gt.data[gt.data[:, 0] == frame_id][:, 2:6]
            for i in range(len(frame_bboxes)):
                for j in range(i+1, len(frame_bboxes)):
                    iou = compute_iou(frame_bboxes[i], frame_bboxes[j])
                    iou_ratios.append(iou)

    return compute_distr_and_avg(iou_ratios, bins=bins)


def compute_iou_ratio_track_inter_frame(gt_tracking: List[TrackingData], bins=5):
    """
    Compute the ratio of the intersection over union (IoU) of the bounding boxes of objects in different frames but in the same track and return a histogram of the ratios and average ratio

    Args:
        gt_tracking: list of tracking ground truth data
        bins: number of bins for the histogram or bin edges

    Returns:
        hist: histogram of the IoU ratios
        bin_edges: bin edges for the histogram
        avg_iou_ratio: average IoU ratio
    """
    iou_ratios = []
    for gt in gt_tracking:
        obj_ids = np.unique(gt.data[:, 1])
        for obj_id in obj_ids:
            obj_data = gt.data[gt.data[:, 1] == obj_id]
            frame_ids = obj_data[:, 0]
            for i in range(len(frame_ids)-1):
                iou = compute_iou(obj_data[i, 2:6], obj_data[i+1, 2:6])
                iou_ratios.append(iou)

    return compute_distr_and_avg(iou_ratios, bins=bins)


def compute_stat_by_name(metric: StatsName):
    """
    Compute the statistic by name

    Args:
        metric: name of the statistic

    Returns:
        hist: histogram of the statistic
        bin_edges: bin edges for the histogram
        avg: average of the statistic
    """
    if metric == StatsName.NUM_OBJ_PER_VIDEO:
        return count_obj_per_video
    elif metric == StatsName.NUM_OBJ_PER_FRAME:
        return count_obj_per_frame
    elif metric == StatsName.VIDEO_LENGTH:
        return compute_video_length
    elif metric == StatsName.TRACK_GAP_LENGTH:
        return compute_track_gap_length
    elif metric == StatsName.IOU_RATIO_OBJECTS_INTRA_FRAME:
        return compute_iou_ratio_objects_intra_frame
    elif metric == StatsName.IOU_RATIO_TRACK_INTER_FRAME:
        return compute_iou_ratio_track_inter_frame
    else:
        raise NotImplementedError(
            "Statistic {} is not implemented".format(metric))
