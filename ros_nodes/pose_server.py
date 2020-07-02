#!/usr/bin/env python

from __future__ import print_function

from question_answering.srv import get_pose, get_poseResponse
import rospy

def handle_get_pose(req):
    print("Returningi")
    return get_poseResponse(4, 5, 2, 0)

def get_pose_server():
    rospy.init_node('pose_server')
    s = rospy.Service('qa_getpose_server', get_pose, handle_get_pose)
    print("pose gogo.")
    rospy.spin()

if __name__ == "__main__":
    get_pose_server()
