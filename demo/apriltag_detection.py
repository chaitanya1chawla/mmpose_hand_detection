#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from pupil_apriltags import Detector


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=12)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
    print("reset done")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)  


    #cap_device = args.device
    #cap_width = args.width
    #cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    # Detector
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 0

    while True:
        start_time = time.time()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        color_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Focal lengths - 
        fx = color_intrin.fx
        fy = color_intrin.fy
        
        # Assuming ppx, ppy (coordinates of the principal point of image, as a pixel offset from the left and top edge)
        # is almost equal to required cx, cy (camera's focal center in pixels). 
        # According to Apriltag documentation - for most cameras this will be approximately the same as the image center
        cx = color_intrin.ppx
        cy = color_intrin.ppy

        #print("fx = {}, fy = {}, ppx = {}, ppy = {}".format(fx,fy,cx,cy))

        image = np.asanyarray(aligned_color_frame.get_data())


        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=0.03,
        )
        
        for tag in tags:
            print("Rotation matrix = \n{}".format(tag.pose_R))
            print("Translation = \n{}".format(tag.pose_t))
            #print("Error = {}".format(tag.pose_err))

        debug_image = draw_tags(debug_image, tags, elapsed_time)

        elapsed_time = time.time() - start_time

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('AprilTag Detect Demo', debug_image)

    #cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        # cv.putText(image,
        #            str(tag_family) + ':' + str(tag_id),
        #            (corner_01[0], corner_01[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
        #            0.6, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()