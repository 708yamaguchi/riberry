#!/usr/bin/env roseus

(load "package://kxreus/euslisp/kxr-interface.l")

(kxr-init)

(defun reset-pose ()
  ;; (send *ri* :angle-vector (send *robot* :angle-vector #f(-90 50 0 50 0 30 0 0 0)))
  ;; (send *ri* :angle-vector (send *robot* :angle-vector #f(50 50 -50 50 0 30 -50 0 0)))
  (send *ri* :angle-vector (send *robot* :angle-vector #f(165 54.1187 -175.869 49.834 0.043945 84.6113 -151.841 0.16876 0.16876)))
  (send *ri* :wait-interpolation))

(defun grasp-can ()
  ;; reach
  (send *ri* :angle-vector (send *robot* :angle-vector #f(109.182 55.2612 -141.899 49.3066 36.3428 66.15 -151.841 -0.13499 -0.13499)))
  (send *ri* :wait-interpolation)
  ;; grasp
  (send *ri* :angle-vector (send *robot* :angle-vector #f(109.182 55.2612 -141.899 49.3066 36.3428 66.15 -151.841 60 -60)))
  (send *ri* :wait-interpolation)
  ;; return
  (send *ri* :angle-vector (send *robot* :angle-vector #f(165 54.1187 -175.869 49.834 0.043945 84.6113 -151.841 60 -60)))
  (send *ri* :wait-interpolation)
  ;; open
  (reset-pose))

(defun cb (msg)
  (when (eq (send msg :data) 1)
    (ros::ros-info "servo-on")
    (send *ri* :servo-on))
  (when (eq (send msg :data) 2)
    (ros::ros-info "servo-off")
    (send *ri* :servo-off))
  (when (eq (send msg :data) 3)
    (ros::ros-info "reset-pose")
    (reset-pose))
  (when (eq (send msg :data) 4)
    (ros::ros-info "grasp-can")
    (grasp-can)))

(defun main ()
  (ros::subscribe "/atom_s3_button_state" std_msgs::Int32 #'cb)
  (ros::spin))

(main)
