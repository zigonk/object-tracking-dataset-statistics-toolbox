import json
import os
import glob
import numpy as np

from type.TrackingType import TrackingData, TrackingQuery


def preprocess_tracking_name(track_name, box_prefix):
    track_name = track_name.split('/')
    if box_prefix not in track_name:
        return ''
    track_name = track_name[track_name.index(box_prefix) + 1:]
    track_name = '/'.join(track_name)
    return track_name


def load_tracking_gt(data_dir: str, box_prefix='box_gt'):
    """
    Load ground truth for tracking data

    Args:
        data_dir: path to the data directory

    Returns:
        gt(List[TrackingData]): list of tracking ground truth data
    """
    gt_files = sorted(glob.glob(os.path.join(
        data_dir, box_prefix, '**/*.txt'), recursive=True))
    gt = []
    for gt_file in gt_files:
        if gt_file == '' or os.path.exists(gt_file) is False:
            continue
        track_name = preprocess_tracking_name(gt_file, box_prefix)
        gt.append(TrackingData(track_name,
                  np.loadtxt(gt_file, delimiter=',')))
    return gt


def load_tracking_query(data_dir: str, box_prefix='box_gt', query_prefix='caption_queries'):
    """
    Load query for tracking data

    Args:
        data_dir: path to the data directory

    Returns:
        query(List[TrackingQuery]): list of tracking query
    """
    query_files = sorted(glob.glob(os.path.join(
        data_dir, query_prefix, '**/*.json'), recursive=True))
    queries = []
    for query_file in query_files:
        data = json.load(open(query_file, 'r'))
        for d in data:
            d['track_path'] = preprocess_tracking_name(
                d['track_path'], box_prefix)
            if 'caption' not in d:
                print('Warning: Caption not found in query {}'.format(d))
            queries.append(TrackingQuery(d))
    return queries


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

    # Append padding for last hist
    hist = np.append(hist, 0)
    bin_edges = np.append(bin_edges, bin_edges[-1])
    # Plot hist align bar title to edge
    plt.bar(bin_edges[:-1], hist, align='edge', width=np.diff(bin_edges))

    # SET UP X AXIS
    plt.xlabel(metric_name)

    # SET UP Y AXIS
    plt.yscale('log')
    plt.ylim(0, 1)
    plt.ylabel('Ratio (log scale)')
    # Y-axis print 2 decimal places
    plt.gca().yaxis.set_major_formatter(
        plt.FormatStrFormatter('%.3f'))

    plt.savefig(os.path.join(output_dir, metric_name), bbox_inches='tight')
    plt.clf()
    plt.close()
