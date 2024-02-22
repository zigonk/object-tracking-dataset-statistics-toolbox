from type.TrackingType import TrackingData, TrackingQuery
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
    bins = min(bins, max(num_objs))
    print("Max num objs per video: {}".format(max(num_objs)))
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
        if len(frame_ids) == 0:
            continue
        for frame_id in frame_ids:
            num_objs.append(len(gt.data[gt.data[:, 0] == frame_id]))
    bins = min(bins, max(num_objs))
    print("Max num objs per frame: {}".format(max(num_objs)))
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

    bins = min(bins, max(video_lengths))
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
    bins = min(bins, max(gap_lengths))
    print("Max gap length: {}".format(max(gap_lengths)))
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
                    if iou > 0:
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


def compute_stat_per_class_name(gt_tracking: List[TrackingData], gt_text_query: List[TrackingQuery]):
    """Compute the number of frames per category

    Args:
        gt_tracking (List[TrackingData]): ground truth tracking data
        gt_text_query (List[TrackingQuery]): ground truth text query data

    Returns:
        num_frames_per_class_name (dict): number of frames per category
        num_objects_per_class_name (dict): number of objects per category
        num_boxes_per_class_name (dict): number of boxes per category
    """
    num_frames_per_class_name = {}
    num_objects_per_class_name = {}
    num_boxes_per_class_name = {}

    for gt in gt_tracking:
        # Check if gt video name in gt_text_query
        index = -1
        for i, gt_query in enumerate(gt_text_query):
            if gt_query.track_path == gt.track_name:
                index = i
                break
        if index == -1:
            print(
                f"Warning: {gt.track_name} not found in gt_text_query. Skipping ...")
            continue

        class_name = gt_text_query[index].class_name
        if class_name not in num_frames_per_class_name:
            num_frames_per_class_name[class_name] = 0
            num_objects_per_class_name[class_name] = 0
            num_boxes_per_class_name[class_name] = 0

        num_objects_per_class_name[class_name] += len(np.unique(gt.data[:, 1]))
        num_boxes_per_class_name[class_name] += len(gt.data)
        num_frames_per_class_name[class_name] += len(np.unique(gt.data[:, 0]))

    return num_frames_per_class_name, num_objects_per_class_name, num_boxes_per_class_name
