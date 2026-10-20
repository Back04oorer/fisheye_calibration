#!/usr/bin/env python3
"""Subscription-only bridge. No publishers, services, SDK or CAN handles."""

import argparse
import json
import time


def main():
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["left", "right"], required=True)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("handeye_feedback_reader")
    topic = f"/slave_{args.side}/joint_states_single"

    def receive(msg):
        value = {
            "names": list(msg.name),
            "positions": list(msg.position),
            "stamp_s": msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            "received_monotonic": time.monotonic(),
            "received_wall": time.time(),
            "topic": topic,
            "frame_id": msg.header.frame_id,
        }
        try:
            print(json.dumps(value, allow_nan=False), flush=True)
        except ValueError:
            pass  # Invalid feedback cannot become a usable pose.

    node.create_subscription(JointState, topic, receive, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
