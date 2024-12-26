#!/usr/bin/env python3

from apriltag_ros.msg import AprilTagCornerDetectionArray
from cameramodels import PinholeCameraModel
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image


# class MarkerDepthPoseEstimator(ConnectionBasedTransport):
class MarkerDepthPoseEstimator:

    def __init__(self):
        super().__init__()

        self.bridge = CvBridge()
        self.camera_info_msg = None
        self.camera_model = None

        self.subscribe()

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        self.actual_marker_size = rospy.get_param(
            '~actual_marker_size', None)  # Unit: [m]
        sub_corner_detections = message_filters.Subscriber(
            '~input/tag_corner_detections',
            AprilTagCornerDetectionArray, queue_size=1)
        sub_depth = message_filters.Subscriber(
            '~input/depth',
            Image, queue_size=1, buff_size=2**24)
        self.subs = [sub_corner_detections, sub_depth]

        self.camera_types = ["color", "depth"]
        self.camera_info_msg = {}
        self.camera_model = {}
        self.sub_info = {}
        for camera_type in self.camera_types:
            self.camera_info_msg[camera_type] = None
            self.camera_model[camera_type] = None
            self.sub_info[camera_type] = rospy.Subscriber(
                '~input/' + camera_type + '/info',
                CameraInfo, self._cb_cam_info, camera_type)

        if rospy.get_param('~approximate_sync', True):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self._cb_with_depth)

    def unsubscribe(self):
        self.sub.unregister()

    def _cb_cam_info(self, msg, camera_type):
        self.camera_info_msg[camera_type] = msg
        self.camera_model[camera_type] = PinholeCameraModel.from_camera_info(
            self.camera_info_msg[camera_type])
        self.sub_info[camera_type].unregister()
        self.sub_info[camera_type] = None
        rospy.loginfo("Received camera info")

    def _cb_with_depth(self, corner_detections_msg, depth_img_msg):
        for camera_type in self.camera_types:
            if self.camera_model[camera_type] is None:
                rospy.logwarn(f'waiting {camera_type} camera info message.')
                return
        bridge = self.bridge
        supported_encodings = {'16UC1', '32FC1'}
        if depth_img_msg.encoding not in supported_encodings:
            rospy.logwarn(
                f'Unsupported depth image encoding: {depth_img_msg.encoding}')

        depth = bridge.imgmsg_to_cv2(depth_img_msg)
        if depth_img_msg.encoding == '16UC1':
            depth = depth / 1000.0  # convert metric: mm -> m

        AprilTagCornerDetectionArray(header=corner_detections_msg.header)
        for detection in corner_detections_msg.detections:
            corners = np.array(detection.corners, dtype=np.int32).reshape(-1, 2)
            corners_in_depth_img = self.batch_transform_pixel(
                self.camera_model["color"],
                self.camera_model["depth"],
                corners)
            corners_in_depth_img = np.round(corners_in_depth_img).astype("int")
            depth_values = depth[tuple(corners_in_depth_img[:, [1, 0]].T)]
            # Corner points calculated in depth image
            xyz = self.camera_model["depth"].batch_project_pixel_to_3d_ray(
                corners_in_depth_img,
                depth_values
            )
            if self.actual_marker_size is not None:
                marker_size_in_depth_img = np.linalg.norm(
                    np.roll(xyz, -1, axis=0) - xyz, axis=1, ord=2)
                tolerance_percent = 0.05
                marker_size_error = self.marker_size_error(
                    marker_size_in_depth_img,
                    self.actual_marker_size)
                if marker_size_error > tolerance_percent:
                    rospy.logerr(f"Marker({detection.id[0]}) recognition fail. " +\
                                 "Marker size error " +\
                                 f"is {marker_size_error*100:.1f}% " +\
                                 f"(> {tolerance_percent*100:.1f}%)"
                    )
                    # For debug
                    # rospy.logerr("Corner points calculated in depth image:\n"+\
                    #              f"{xyz}")
                    # rospy.logerr(
                    #     f"Corner to Corner distances: {marker_size_in_depth_img}")
                    return
                else:
                    rospy.loginfo(f"Marker({detection.id[0]}) recognition succeed. " +\
                                  "Marker size error " +\
                                  f"is {marker_size_error*100:.1f}% " +\
                                  f"(< {tolerance_percent*100:.1f}%)"
                    )
            rospy.logerr(xyz)
            # See
            # https://github.com/AprilRobotics/apriltag/blob/fc2a7b20a49fc8b211709280e337a76a30db3042/apriltag.c#L980
            # https://github.com/iory/apriltag_ros/blob/e0e995b8c1b326c060339b330423ecc199fa60a5/apriltag_ros/src/common_functions.cpp#L631
            # xyz order: [1, -1], [1, 1], [-1, 1], [-1, -1]
            marker_pose_with_depth = self.calculate_pose(xyz)
            rospy.logerr(marker_pose_with_depth)

    def batch_transform_pixel(self, camera1, camera2, pixels1):
        """
        Batch transform pixel coordinates from camera1 to camera2.

        Parameters:
        - camera1: PinholeCameraModel instance for the first camera.
        - camera2: PinholeCameraModel instance for the second camera.
        - pixels1: numpy.ndarray of shape (N, 2), where N is the number of pixels.
                   Each row represents a pixel coordinate (x1, y1) in camera1.

        Returns:
        - pixels2: numpy.ndarray of shape (N, 2), representing the corresponding
                   pixel coordinates (x2, y2) in camera2.
        """
        # Step 1: Project the batch of pixels from camera1 to 3D rays
        rays = camera1.batch_project_pixel_to_3d_ray(pixels1)

        # Step 2: Assume all points lie on the image plane (z=1)
        points_3d = rays * np.array([1, 1, 1])

        # Step 3: Project the 3D points onto camera2's image plane
        pixels2 = camera2.batch_project3d_to_pixel(points_3d)

        return pixels2

    def marker_size_error(self, arr, marker_size):
        """
        Check if any array element is outside marker_size ± tolerance range

        Args:
            arr: numpy array of values to check. Unit is [m].
            marker_size: Actual marker size. Unit is [m].

        Returns:
            float: Percentage of error
        """
        lower_error = 1.0 - np.min(arr) / float(marker_size)
        upper_error = np.max(arr) / float(marker_size) - 1.0
        return max(lower_error, upper_error)

    def calculate_pose(self, points):
        """
        Calculate the pose based on the given points and specified coordinate system.

        Args:
            points (np.ndarray): A 4x3 numpy array representing the coordinates of the points.

        Returns:
            Pose: A geometry_msgs.msg.Pose object representing the calculated pose.
        """
        # Points from the array
        p1 = points[1]  # 2nd point (index 1)
        p2 = points[2]  # 3rd point (index 2)
        p3 = points[3]  # 4th point (index 3)

        # Set the origin to the 3rd point
        origin = p2

        # Compute z-axis as the direction from p3 to p4
        z_dir = p3 - p2
        z_dir /= np.linalg.norm(z_dir)  # Normalize

        # Compute y-axis as the direction from p1 to p2
        y_dir = p1 - p2
        y_dir /= np.linalg.norm(y_dir)  # Normalize

        # Compute x-axis as the cross product of y-axis and z-axis
        x_dir = np.cross(y_dir, z_dir)
        x_dir /= np.linalg.norm(x_dir)  # Normalize

        # Recompute z-axis to ensure orthogonality
        z_dir = np.cross(x_dir, y_dir)
        z_dir /= np.linalg.norm(z_dir)

        # Construct the rotation matrix
        rotation_matrix = np.array([x_dir, y_dir, z_dir]).T

        # Convert the rotation matrix to a quaternion
        q = self.rotation_matrix_to_quaternion(rotation_matrix)

        # Create and populate the Pose message
        pose = Pose()
        pose.position.x = origin[0]
        pose.position.y = origin[1]
        pose.position.z = origin[2]
        pose.orientation = Quaternion(*q)

        return pose

    def rotation_matrix_to_quaternion(self, matrix):
        """
        Convert a rotation matrix to a quaternion.

        Args:
            matrix (np.ndarray): A 3x3 rotation matrix.

        Returns:
            tuple: A quaternion (x, y, z, w).
        """
        m = matrix
        t = np.trace(m)
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return x, y, z, w

if __name__ == '__main__':
    rospy.init_node('marker_depth_pose_estimator')
    node = MarkerDepthPoseEstimator()
    rospy.spin()
