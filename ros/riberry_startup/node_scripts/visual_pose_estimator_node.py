#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from riberry.visual_pose_estimator import VisualPoseEstimator

if __name__ == "__main__":
    try:
        rospy.init_node("visual_pose_estimator")
        VisualPoseEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
