# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
import copy
import numpy as np
from argparse import ArgumentParser
#from IPython import embed 

import pyrealsense2 as rs
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R

from coordinate_transformation import transform_point_3d_from_cam_to_ground, transform_pose_from_cam_to_ground
# using bindings.py from pupil_apriltags
from pupil_apriltags import Detector
import json_tricks as json
import mmcv
import mmengine
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False



## The following data is being saved --
#   
#   1. Frame index
#   2. 21 Keypoints of a single hand in 3d wrt ground tag per frame
#   3. Average score of all keypoints per frame
#   4. List saying whether keypoints are valid or not
#   5. April tags per frame--
#       i.   tag_id
#       ii.  position wrt ground tag
#       iii. quaternions wrt ground tag
#       iv.  pose error
#   6. Primary Camera for entire trajectory



def get_args():
    parser = ArgumentParser()

    parser.add_argument("--task_name", type=str, required=True)

    # AprilTag required arguments - 
    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--primary_camera", type=int, help='specify if camera 0 or 1 was the primary camera', required=True)
    parser.add_argument("--primary_keypoint_camera", type=int, help='specify if camera 0 or 1 was the primary camera for detecting hand pose', required=True)
    parser.add_argument("--expected_tags", nargs='+', type=int, help='specify tag ids being used in the task demonstration', required=True)

    # MMPose required arguments - 
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--input', type=str, default='webcam', help='Image/Video file')
    parser.add_argument('--show', action='store_true', default=True, help='whether to show img')
    parser.add_argument('--output-root', type=str, default='', help='root of the output img file. Default not saving the visualization images.')
    parser.add_argument('--output_data_folder', type=str, default='', help='Folder for saving recorded numpy files.', required=True)
    parser.add_argument('--save_predictions', action='store_true', default=True, help='whether to save predicted results. Keep always True')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--det-cat-id', type=int, default=0, help='Category id for bounding box detection model')
    parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.3, help='IoU threshold for bounding box NMS')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Visualizing keypoint thresholds. Keypoints with confidence below this would not be drawn')
    parser.add_argument('--draw-heatmap', action='store_true', default=False, help='Draw heatmap predicted by the model')
    parser.add_argument('--show-kpt-idx', action='store_true', default=False, help='Whether to show the index of keypoints')
    parser.add_argument('--skeleton-style', default='mmpose', type=str, choices=['mmpose', 'openpose'], help='Skeleton style selection')
    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument('--show_tags', action='store_true', default=False, help='Show detected AprilTags')

    args = parser.parse_args()

    return args

def get_avg_score(scores):
    return sum(scores)/len(scores)
        
def get_depth(keypoints, depth_image, depth_intrin, cam_num, valid_kpt_frames ):

    valid_kpts = True
    kpts_3d = []
    index_finger_tip = keypoints[8]
    print(index_finger_tip)
    #depth1 = aligned_depth_frame.get_distance( int(index_finger_tip[1]),int(index_finger_tip[0]) )
    
    if not (int(index_finger_tip[1]) > 720 or int(index_finger_tip[0]) > 1280):
        depth2 = depth_image[int(index_finger_tip[1]),int(index_finger_tip[0])]
        print("CAM{} | depth2 at index_finger_tip = ".format(cam_num), depth_image[int(index_finger_tip[1]),int(index_finger_tip[0])])
        depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(
                                                depth_intrin, 
                                                [int(index_finger_tip[0]),int(index_finger_tip[1])],
                                                depth2/1000)
        print("CAM{} | camera coord = ".format(cam_num), depth_point_in_meters_camera_coords)
    
    for kpt in keypoints:
        if int(kpt[0]) > 1280 or int(kpt[1]) > 720:
            valid_kpts=False
            continue

        depth = depth_image[int(kpt[1]),int(kpt[0])]
        kpts_3d.append(rs.rs2_deproject_pixel_to_point(
                                                    depth_intrin, 
                                                    [int(kpt[0]),int(kpt[1])],
                                                    depth/1000))

    if np.all(np.array(kpts_3d)==0.0):
        valid_kpts=False

    if valid_kpts:
        valid_kpt_frames['cam{}'.format(cam_num)].append(1)
    else:
        valid_kpt_frames['cam{}'.format(cam_num)].append(0)

    return kpts_3d

def init_camera():

    print("starting reset")
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    pipelines = []

    for dev in devices:
        dev.hardware_reset()
        serials.append(dev.get_info(rs.camera_info.serial_number))
    print("reset done")
    
    serials = serials[-2:]
    serials.reverse()
    # 821212062747 -- cam0 --- on the left 
    # 821212061298 -- cam1 --- on the right
    print("selected cameras - ", serials)

    for serial in serials:
        
        # Configure depth and color streams
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        # Start streaming
        pipe.start(config)
        pipelines.append(pipe)


        # Get device product line for setting a supporting resolution
        #pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        #pipeline_profile = config.resolve(pipeline_wrapper)
        #device = pipeline_profile.get_device()
        #device_product_line = str(device.get_info(rs.camera_info.product_line))

        #found_rgb = False
        #for s in device.sensors:
        #    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        #        found_rgb = True
        #        break
        #if not found_rgb:
        #    print("The demo requires Depth camera with Color sensor")
        #    exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipelines, align

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)

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

        cv2.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv2.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv2.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        # cv2.putText(image,
        #            str(tag_family) + ':' + str(tag_id),
        #            (corner_01[0], corner_01[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
        #            0.6, (0, 255, 0), 1, cv.LINE_AA)
        cv2.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv2.LINE_AA)

    return image

def pose_detection(args, detector, pose_estimator, visualizer, frame_idx,
                   pipelines, align, cam_intrinsics, data, output_file):
    
    start_time = time.time()
    for cam_num, pipe in enumerate(pipelines):
    
        # Wait for a coherent pair of frames: depth and color
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not aligned_color_frame:
            continue
    
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        debug_image = copy.deepcopy(color_image)
        #gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        if not cam_intrinsics['saved{}'.format(cam_num)]:
            cam_intrinsics['cam{}'.format(cam_num)] = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            cam_intrinsics['saved{}'.format(cam_num)]=True

        # topdown pose estimation
        pred_instances = process_one_image(args, color_image, detector,
                                          pose_estimator, visualizer, 
                                           0.001)
    
        if args.save_predictions:

            # save prediction results
            first_hand = split_instances(pred_instances)[0]

            kpts_3d = get_depth( first_hand['keypoints'], depth_image, depth_intrin, cam_num, data['valid_keypoint_frames'] )
            avg_score = get_avg_score( first_hand['keypoint_scores'] )
            print("CAM{} | avg_score = ".format(cam_num), avg_score)
            
            #embed()

            data['images']['cam{}'.format(int(cam_num))].append(debug_image)
            data['raw_keypoints']['cam{}'.format(int(cam_num))].append(kpts_3d)
            data['avg_keypoint_score']['cam{}'.format(int(cam_num))].append(avg_score)

            k1 = np.array(kpts_3d[0]) # base of the wrist
            k2 = np.array(kpts_3d[9]) # base of the middle finger

            orientation = np.append(k1-k2, 0.)
            data['raw_hand_orientation']['cam{}'.format(int(cam_num))].append(orientation)
            #data['tag_detections']['cam{}'.format(int(cam_num))].append(tag_detections)
            
        # # show tags
        # if args.show_tags:
        #     debug_image = draw_tags(debug_image, tags, elapsed_time)
        #     elapsed_time = time.time() - start_time
        #     key = cv2.waitKey(1)
        #     if key == 27:  # ESC
        #         break
        #     cv2.imshow('AprilTag Detect Demo', debug_image)

        # # output videos
        # if output_file:
        #     frame_vis = visualizer.get_image()
        #     if video_writer is None:
        #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #         # the size of the image with visualization may vary
        #         # depending on the presence of heatmaps
        #         video_writer = cv2.VideoWriter(
        #             output_file,
        #             fourcc,
        #             25,  # saved fps
        #             (frame_vis.shape[1], frame_vis.shape[0]))
        #     video_writer.write(mmcv.rgb2bgr(frame_vis))

        if args.show:
            # press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
                    
            time.sleep(args.show_interval)

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = get_args()

    assert args.task_name != '' # Please give a task_name
    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    at_detector = Detector(families=families, nthreads=nthreads, quad_decimate=quad_decimate,
                           quad_sigma=quad_sigma, refine_edges=refine_edges, 
                           decode_sharpening=decode_sharpening, debug=debug,
                           )

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    #if args.save_predictions:
    #    assert args.output_root != ''
    #    args.pred_save_path = f'{args.output_root}/results_' \
    #        f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)    

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer, 0)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            pipelines, align = init_camera()
            # cap = cv2.VideoCapture(8)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []

        data = {
            'frame_id': [], 
            'raw_keypoints': {'cam0':[], 'cam1':[]}, 
            'avg_keypoint_score': {'cam0':[], 'cam1':[]}, 
            'valid_keypoint_frames': {'cam0':[], 'cam1':[]},
            'raw_hand_orientation': {'cam0':[], 'cam1':[]},
            'tag_detections_in_cam': {'cam0':[], 'cam1':[]},
            'primary_camera': args.primary_camera,
            'primary_keypoint_camera':args.primary_keypoint_camera,
            'expected_tags':args.expected_tags, # list of ints
            'images': {'cam0':[], 'cam1':[]}
            }
        
        ##################
        # Expected objects and corresponding tag ids-
        # Box - 1,2
        # Drawer - 1,2
        # White Bowl - 2
        # Stirrer - 3
        # Red Cup - 6
        ##################
        
        cam_intrinsics = { }
        cam_intrinsics['saved0']=False
        cam_intrinsics['saved1']=False
        frame_idx = 0

        
        ##while cap.isOpened():
        while True:
            try:
                data['frame_id'].append(frame_idx)
                pose_detection(args, detector, pose_estimator, visualizer, frame_idx,
                               pipelines, align, cam_intrinsics, data, output_file)
                
                frame_idx += 1
                if frame_idx == 2:
                    start_time = time.time()
            
            except RuntimeError as e:
                if e.args[0] == 'RuntimeError: out of range value for argument "y"':
                    continue
            
            except KeyboardInterrupt:
                if video_writer:
                    video_writer.release()

                cv2.destroyAllWindows()

                for pipe in pipelines:
                    pipe.stop()
                break
        
        # It could be possible that length of frames from cam0 are 1 more than cam1, depending on 
        # when user pressed ctrl+c
        if len(data['raw_keypoints']['cam0']) != len(data['raw_keypoints']['cam1']):
            
            # Reduce length of all cam0 arrays by 1
            data['raw_keypoints']['cam0'] = data['raw_keypoints']['cam0'][:-1]
            data['raw_hand_orientation']['cam0'] = data['raw_hand_orientation']['cam0'][:-1]
            data['avg_keypoint_score']['cam0'] = data['avg_keypoint_score']['cam0'][:-1]
            data['valid_keypoint_frames']['cam0'] = data['valid_keypoint_frames']['cam0'][:-1]
            data['images']['cam0'] = data['images']['cam0'][:-1]

        elapsed_time = time.time() - start_time
        print('time recorded = ', elapsed_time)
        print('frames recorded per camera = ', len(data['frame_id']))
        print('frequency = ', len(data['frame_id'])/elapsed_time)
        ################
        # Implementing tag detection on saved images now - 
        ################
        print("####### Implementing tag detection on saved images now #######")

        for cam in ['cam0', 'cam1']:
            
            color_intrin = cam_intrinsics[cam]
            for img in data['images'][cam]:
                gray_image = copy.deepcopy(img)
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
                # Focal lengths - 
                fx = color_intrin.fx
                fy = color_intrin.fy
                # Assuming ppx, ppy (coordinates of the principal point of image, as a pixel offset from the left and top edge)
                # is almost equal to required cx, cy (camera's focal center in pixels). 
                # According to Apriltag documentation - for most cameras this will be approximately the same as the image center
                cx = color_intrin.ppx
                cy = color_intrin.ppy
                tags = at_detector.detect(gray_image, estimate_tag_pose=True,
                                          camera_params=[fx, fy, cx, cy], 
                                          tag_size=0.03,)
                tag_list=[]
                for tag in tags:
                    tag_list.append(
                        dict(
                            tag_id=tag.tag_id,
                            position=tag.pose_t.reshape(3,),
                            orientation=R.from_matrix(tag.pose_R).as_quat(),
                            pose_err=tag.pose_err
                        ))    
                data['tag_detections_in_cam'][cam].append(tag_list)


    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        assert args.output_data_folder != ''
        directory = os.path.join(args.output_data_folder, args.task_name)
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        curr_time = time.strftime("%d%m%Y_%H%M%S", time.localtime())
        file_path = os.path.join(directory, "demo_"+str(curr_time)+".npy")
        with open(file_path, "wb") as file:
            name_and_data = (args.task_name, data)
            np.save(file, name_and_data)
        print('predictions have been saved at {}'.format(file_path))

    if output_file:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
