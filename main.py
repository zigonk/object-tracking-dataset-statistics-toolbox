import argparse
from type.StatsNameEnum import StatsName

from utils.io import load_tracking_gt, plot_hist
from utils.stats_tool import (
    compute_stat_by_name
)


def main():
    parser = argparse.ArgumentParser(
        description='Data statistics for object tracking')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory', default='dataset')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory', default='output')

    args = parser.parse_args()

    gt = load_tracking_gt(args.data_dir)

    # Compute statistics
    avg_values = {}
    print("Computing statistics ...")
    stats_eval = []
    stats_eval.append({'name': StatsName.NUM_OBJ_PER_VIDEO, 'bins': 5})
    stats_eval.append({'name': StatsName.NUM_OBJ_PER_FRAME, 'bins': 5})
    stats_eval.append(
        {'name': StatsName.VIDEO_LENGTH, 'bins': [0, 100, 200, 300, 10000]})
    stats_eval.append({'name': StatsName.TRACK_GAP_LENGTH,
                      'bins': [0, 5, 10, 20, 40, 10000]})
    stats_eval.append(
        {'name': StatsName.IOU_RATIO_OBJECTS_INTRA_FRAME, 'bins': 5})
    stats_eval.append(
        {'name': StatsName.IOU_RATIO_TRACK_INTER_FRAME, 'bins': 5})

    for stat in stats_eval:
        hist, bin_edges, avg = compute_stat_by_name(
            stat['name'])(gt, stat['bins'])
        plot_hist(stat['name'].value, hist, bin_edges, args.output_dir)
        avg_values[stat['name'].value] = avg
        print("Average {}: {}".format(stat['name'].value, avg))

    # Save average values to a file
    print("Saving average values to a csv ...")
    with open('output/avg_values.csv', 'w') as f:
        for key in avg_values.keys():
            f.write("Avg %s,%s\n" % (key, avg_values[key]))

    print("Done")


if __name__ == "__main__":
    main()
