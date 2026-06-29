#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import rosbag
import cv2
from cv_bridge import CvBridge

def main():
    parser = argparse.ArgumentParser(
        description='Extract debug and raw images from a ROS bag matching their timestamps.'
    )
    parser.add_argument('bag_path', type=str, help='Path to the input ROS bag file')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment (used in filenames)')
    parser.add_argument(
        '-o', '--output-dir', type=str, default='.',
        help='Directory where the extracted images will be saved (default: current directory)'
    )
    parser.add_argument(
        '--postfix', action='store_true',
        help='Attach the experiment name as a postfix instead of a prefix'
    )
    
    args = parser.parse_args()

    bag_path = args.bag_path
    experiment_name = args.experiment_name
    output_dir = args.output_dir
    use_postfix = args.postfix

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Opening ROS bag: {bag_path}")
    try:
        bag = rosbag.Bag(bag_path)
    except Exception as e:
        print(f"Error opening ROS bag: {e}", file=sys.stderr)
        sys.exit(1)

    debug_topic = '/visual_pose_estimator/debug_image'
    raw_topic = '/decompressed/camera/color/image_raw'

    debug_msgs = []
    raw_msgs_by_stamp = {}

    print("Reading messages from bag...")
    try:
        for topic, msg, t in bag.read_messages(topics=[debug_topic, raw_topic]):
            if topic == debug_topic:
                debug_msgs.append(msg)
            elif topic == raw_topic:
                raw_msgs_by_stamp[msg.header.stamp] = msg
    except Exception as e:
        print(f"Error reading messages from bag: {e}", file=sys.stderr)
        bag.close()
        sys.exit(1)
    finally:
        bag.close()

    num_debug = len(debug_msgs)
    print(f"Found {num_debug} messages on {debug_topic}")
    print(f"Found {len(raw_msgs_by_stamp)} messages on {raw_topic}")

    if num_debug == 0:
        print(f"Error: No messages found on topic {debug_topic}", file=sys.stderr)
        sys.exit(1)
    elif num_debug >= 2:
        print(
            f"\n[WARNING] The topic '{debug_topic}' contains {num_debug} messages (2 or more).\n"
            "An image pair will be saved for each message.",
            file=sys.stderr
        )

    bridge = CvBridge()

    for i, debug_msg in enumerate(debug_msgs):
        stamp = debug_msg.header.stamp
        stamp_str = f"{stamp.secs}_{stamp.nsecs}"
        print(f"\nProcessing message [{i+1}/{num_debug}] with stamp {stamp.to_sec()}...")

        # Convert debug image
        try:
            cv_debug_img = bridge.imgmsg_to_cv2(debug_msg, desired_encoding="bgr8")
        except Exception as e:
            print(f"  Error converting debug image: {e}", file=sys.stderr)
            continue

        # Find matching raw image
        cv_raw_img = None
        raw_msg = raw_msgs_by_stamp.get(stamp)
        
        if raw_msg is not None:
            print("  Exact timestamp match found for raw image.")
            try:
                cv_raw_img = bridge.imgmsg_to_cv2(raw_msg, desired_encoding="bgr8")
            except Exception as e:
                print(f"  Error converting raw image: {e}", file=sys.stderr)
        else:
            print("  [WARNING] No exact timestamp match found in raw images.", file=sys.stderr)
            if raw_msgs_by_stamp:
                # Find closest match
                closest_stamp = min(raw_msgs_by_stamp.keys(), key=lambda s: abs((s - stamp).to_sec()))
                diff = abs((closest_stamp - stamp).to_sec())
                print(f"  Using closest raw image (stamp: {closest_stamp.to_sec()}, diff: {diff:.6f}s)")
                
                raw_msg = raw_msgs_by_stamp[closest_stamp]
                try:
                    cv_raw_img = bridge.imgmsg_to_cv2(raw_msg, desired_encoding="bgr8")
                except Exception as e:
                    print(f"  Error converting closest raw image: {e}", file=sys.stderr)
            else:
                print("  No raw images available in the bag to match.", file=sys.stderr)

        # Determine filenames
        # If there are multiple debug images, append an index to avoid overwriting.
        suffix = f"_{i}" if num_debug > 1 else ""
        if use_postfix:
            debug_filename = f"recognition{suffix}_{experiment_name}.png"
            raw_filename = f"raw{suffix}_{experiment_name}.png"
        else:
            debug_filename = f"{experiment_name}_recognition{suffix}.png"
            raw_filename = f"{experiment_name}_raw{suffix}.png"

        debug_path = os.path.join(output_dir, debug_filename)
        raw_path = os.path.join(output_dir, raw_filename)

        # Save debug image
        try:
            cv2.imwrite(debug_path, cv_debug_img)
            print(f"  Saved debug image to: {debug_path}")
        except Exception as e:
            print(f"  Error saving debug image: {e}", file=sys.stderr)

        # Save raw image
        if cv_raw_img is not None:
            try:
                cv2.imwrite(raw_path, cv_raw_img)
                print(f"  Saved raw image to: {raw_path}")
            except Exception as e:
                print(f"  Error saving raw image: {e}", file=sys.stderr)

    print("\nExtraction complete.")

if __name__ == '__main__':
    main()
