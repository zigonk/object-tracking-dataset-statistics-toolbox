import argparse
import os

from type.StatsNameEnum import StatsName
from utils.io import load_tracking_gt, plot_hist
from utils.tracking_stats_tool import compute_stat_by_name


def main():
    parser = argparse.ArgumentParser(
        description='Data statistics for object tracking')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory', default='dataset')
    parser.add_argument('--box_prefix', type=str, default='box_gt',
                        help='Prefix for the bounding box ground truth files')
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset name, empty string for all datasets')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory', default='output')

    args = parser.parse_args()

    # Construct path for dataset
    if args.dataset == '':
        box_prefix = args.box_prefix
        output_dir = args.output_dir
    else:
        box_prefix = args.box_prefix + '/' + args.dataset
        output_dir = args.output_dir + '/' + args.dataset

    os.makedirs(output_dir, exist_ok=True)

    gt = load_tracking_gt(args.data_dir, box_prefix)

    # Compute statistics
    avg_values = {}
    print("Computing statistics ...")
    stats_eval = []
    stats_eval.append({'name': StatsName.NUM_OBJ_PER_VIDEO, 'bins': 10})
    stats_eval.append({'name': StatsName.NUM_OBJ_PER_FRAME, 'bins': 10})
    stats_eval.append(
        {'name': StatsName.VIDEO_LENGTH, 'bins': 10})
    stats_eval.append({'name': StatsName.TRACK_GAP_LENGTH,
                      'bins': 10})
    stats_eval.append(
        {'name': StatsName.IOU_RATIO_OBJECTS_INTRA_FRAME, 'bins': 10})
    stats_eval.append(
        {'name': StatsName.IOU_RATIO_TRACK_INTER_FRAME, 'bins': 10})

    # Remove csv file if it exists
    if os.path.exists(f'{output_dir}/hist_values.csv'):
        os.remove(f'{output_dir}/hist_values.csv')

    for stat in stats_eval:
        hist, bin_edges, avg = compute_stat_by_name(
            stat['name'])(gt, stat['bins'])
        avg_values[stat['name'].value] = avg
        print("Average {}: {}".format(stat['name'].value, avg))
        hist_csv = ""
        for i in range(len(hist)):
            # Set precision to 3 decimal places
            hist_csv += "%.3f," % hist[i]
        hist_csv = hist_csv[:-1]
        bin_edges_csv = ""
        for i in range(len(bin_edges)):
            # Set precision to 3 decimal places
            bin_edges_csv += "%.3f," % bin_edges[i]
        with open(f'{output_dir}/hist_values.csv', 'a') as f:
            f.write("Hist %s,%s\n" % (stat['name'].value, hist_csv))
            f.write("Bin edges %s,%s\n" %
                    (stat['name'].value, bin_edges_csv))

        plot_hist(stat['name'].value, hist, bin_edges, output_dir)

    # Save average values to a file
    print("Saving average values to a csv ...")
    with open(f'{output_dir}/avg_values.csv', 'w') as f:
        for key in avg_values.keys():
            f.write("Avg %s,%s\n" % (key, avg_values[key]))

    print("Done")


if __name__ == "__main__":
    main()
