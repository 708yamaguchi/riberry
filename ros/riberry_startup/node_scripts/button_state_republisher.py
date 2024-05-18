#!/usr/bin/env python3

import rospy

from std_msgs.msg import Int32, Empty

pub_toggle = None

def callback(msg):
    global pub_toggle
    if msg.data == 11:
        pub_toggle.publish(Empty())        

if __name__ == '__main__':
    rospy.init_node('button_state_publisher')
    pub_toggle = rospy.Publisher("/robot_a/vacuum_toggle", Empty, queue_size=1)
    sub = rospy.Subscriber("/atom_s3_button_state", Int32, callback=callback, queue_size=1)
    rospy.spin()
