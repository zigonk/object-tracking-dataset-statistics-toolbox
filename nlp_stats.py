import argparse
import os

import numpy as np

from utils.io import load_tracking_gt, load_tracking_query
from utils.textual_stats_tool import (build_word_cloud,
                                      count_avg_sentence_length,
                                      preprocess_text, unique_word_count)
from utils.tracking_stats_tool import (compute_stat_by_name,
                                       compute_stat_per_class_name)


def main():
    parser = argparse.ArgumentParser(
        description='Data statistics for object tracking')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory', default='dataset')
    parser.add_argument('--box_prefix', type=str, default='box_gt',
                        help='Prefix for the bounding box ground truth files')
    parser.add_argument('--query_prefix', type=str, default='caption_queries',
                        help='Prefix for the query files')
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset name, empty string for all datasets')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory', default='output')

    args = parser.parse_args()

    # Load data
    # Construct path for dataset
    if args.dataset == '':
        box_prefix = args.box_prefix
        output_dir = args.output_dir
        query_prefix = args.query_prefix
    else:
        box_prefix = args.box_prefix + '/' + args.dataset
        query_prefix = args.query_prefix + '/' + args.dataset
        output_dir = args.output_dir + '/' + args.dataset

    os.makedirs(output_dir, exist_ok=True)

    gt_tracking = load_tracking_gt(args.data_dir, box_prefix)
    gt_text = load_tracking_query(args.data_dir, box_prefix, query_prefix)

    # Compute statistics
    print("Computing statistics ...")
    # Count word in caption, definition and attributes, synonyms
    list_fields = ['caption', 'definition', 'attributes', 'synonyms', 'type']
    f = open(f"{output_dir}/unique_word_count.csv", "w")
    f.write("field, count\n")
    for field in list_fields:
        count, summary, repeats = unique_word_count(gt_text, field)
        f.write(f"{field}, {count}\n")
        # Write summary and repeats to csv file
        f_field = open(f"{output_dir}/unique_word_count_{field}.csv", "w")
        f_field.write("word, count\n")
        for i in range(len(summary)):
            f_field.write(f"{summary[i]}, {repeats[i]}\n")
        f_field.write(f"Total, {np.sum(repeats)}\n")
        f_field.close()
    f.close()

    # Count average len of caption, definition, attributes, synonyms
    f = open(f"{output_dir}/avg_len.csv", "w")
    f.write("field, avg_len\n")
    for field in list_fields:
        avg_len = count_avg_sentence_length(gt_text, field)
        f.write(f"{field}, {avg_len}\n")
    f.close()

    # Count frames, bounding boxes, objects per class name
    num_frames, num_objects, num_boxes = compute_stat_per_class_name(
        gt_tracking, gt_text)

    f = open(f"{output_dir}/stats_of_class_name.csv", "w")
    f.write("class_name, num_frames, num_objects, num_boxes\n")
    for class_name in num_frames:
        f.write(
            f"{class_name}, {num_frames[class_name]}, {num_objects[class_name]}, {num_boxes[class_name]}\n")
    f.close()


if __name__ == '__main__':
    main()
