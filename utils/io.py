import os
import glob
import numpy as np

from types.TrackingType import TrackingData

def load_tracking_gt(data_dir: str):
    """
    Load ground truth for tracking data

    Args:
        data_dir: path to the data directory

    Returns:
        gt(List[TrackingData]): list of tracking ground truth data
    """
    gt_files = sorted(glob.glob(os.path.join(data_dir,'**/*.txt'), recursive=True))
    gt = []
    for gt_file in gt_files:
        gt.append(TrackingData(os.path.basename(gt_file).split('.')[0], np.loadtxt(gt_file, delimiter=',')))
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
    plt.hist(hist, bins=bin_edges)
    plt.title(metric_name)
    plt.savefig(os.path.join(output_dir, metric_name+'.png'))
    plt.close()
        
