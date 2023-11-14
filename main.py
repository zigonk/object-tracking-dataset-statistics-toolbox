import argparse
from types.StatsNameEnum import StatsName

from utils.io import load_tracking_gt, plot_hist
from utils.stats_tool import (
    compute_stat_by_name
)


def main():
    parser = argparse.ArgumentParser(description='Data statistics for object tracking')
    parser.add_argument('--data_dir', type=str, help='Path to the data directory', default='data/MOT17/train')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory', default='output')
    
    args = parser.parse_args()

    gt = load_tracking_gt(args.data_dir)

    # Compute statistics
    avg_values = {}
    print("Computing statistics ...")
    stat_list = [StatsName.NUM_OBJ_PER_VIDEO, 
                 StatsName.NUM_OBJ_PER_FRAME, 
                 StatsName.VIDEO_LENGTH, 
                 StatsName.TRACK_GAP_LENGTH, 
                 StatsName.IOU_RATIO_OBJECTS_INTRA_FRAME, 
                 StatsName.IOU_RATIO_TRACK_INTER_FRAME]
    
    for stat in stat_list:
        hist, bin_edges, avg = compute_stat_by_name(stat)(gt)
        plot_hist(stat.value, hist, bin_edges, args.output_dir)
        avg_values[stat.value] = avg
        print("Average {}: {}".format(stat.value, avg))
    
    # Save average values to a file
    print("Saving average values to a csv ...")
    with open('output/avg_values.csv', 'w') as f:
        for key in avg_values.keys():
            f.write("Avg %s,%s\n"%(key,avg_values[key]))

    print("Done")       

if __name__ == "__main__":
    main()