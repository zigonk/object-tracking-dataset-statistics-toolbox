import os
import glob
import numpy as np

from type.TrackingType import TrackingData


def load_tracking_gt(data_dir: str):
    """
    Load ground truth for tracking data

    Args:
        data_dir: path to the data directory

    Returns:
        gt(List[TrackingData]): list of tracking ground truth data
    """
    gt_files = sorted(glob.glob(os.path.join(
        data_dir, '**/*.txt'), recursive=True))
    gt = []
    for gt_file in gt_files:
        gt.append(TrackingData(os.path.basename(gt_file).split(
            '.')[0], np.loadtxt(gt_file, delimiter=',')))
    return gt


def plot_hist(metric_name, hist, bin_edges, output_dir):
    """
    Plotting the statistic information to a file

    Args:
        metric_name: name of the metric
        hist: histogram of the metric
        bin_edges: bin edges of the histogram
        output_dir: path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    # Plot histogram from output of numpy historgram by using plt bar and algin the bin edges
    plt.bar(bin_edges[:-1], hist, width=bin_edges[1] -
            bin_edges[0], align='edge')
    plt.title(metric_name)
    plt.xlabel(metric_name)
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, metric_name+".png"))
    plt.clf()
    plt.close()
