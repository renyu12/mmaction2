# Superstar Simulator APP
# Framework from openmmlab (mmpose webcam demo + mmaction2 webcam demo)
# NTU MLDA developed
#     Created by Renyu 24.6.8

# renyu: prerequisite: 
#            install mmpose and mmaction2(must from source code)
#            put the script in mmaction2/Superstar
#            download pretrained models and get the pretrained models config file (should be in mmpose/mmaction2 source code) to localpath (remember to set the path the start command)
# renyu: run the script use command:
#        python webcam_skeletion_action.py [human detection model config] [human detection model path] [pose estimation model config] [pose estimation model path] [action recognition model config] [action recognition model path] [action label path] --other_args, for example:
#        python webcam_skeleton_action.py rtmdet_nano_320-8xb32_coco-person.py rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth rtmpose-m_8xb256-420e_coco-256x192.py rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth tsn_r50_1x1x8_video_infer.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth label_map_k400.txt --average-size 5 --threshold 0.2 --show
#        if not gpu, use --device cpu, but it might not be able to run smoothly for big models...

import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.utils import get_str_type

# renyu: from mmpose demo
import mmcv
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

# for random select music
import random

# renyu: sould lib for Windows (other platform may need to change it)
#        set the the pair of (action name, sound file) here
#        (can't play 2 music together, deprecated)
#import winsound
import pygame
SOUND_DICT = {
    'beatboxing': 'happy.wav'
}

# renyu: play different kinds of music according to the number of people
#        randomly choose from the list
BGM_DICT = {
    0 : ['0after_the_rain.mp3'],
    1 : ['1cafe_train.mp3'],
    2 : ['2background_dreams.mp3'],
    3 : ['3be_happy.mp3','3lovely_sunny_day.mp3']
}

# renyu: add meme picture on the screen
PICTURE_DICT = {
    'drinking': 'spillTea.jpg',
    'drinking beer': 'spillTea.jpg',
    'drinking shots': 'spillTea.jpg',
    'beatboxing': "big.jpg"
}

# renyu: set the camera resolution, but not all the resolution can be supported, only several levels are valid
#        check the real resolution according to the print log
#        (resolution might affect the performance...)
CAMERAWIDTH = 960
CAMERAHEIGHT = 540

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
SPECIALFONTCOLOR = (0, 0, 255)  # BGR, red, for making sound actions
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

# renyu: all the supported start args here
def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')

    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')

    parser.add_argument('action_config', help='test config file path')
    parser.add_argument('action_checkpoint', help='checkpoint file/url')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--bad-computer',
        action='store_true',
        default=False,
        help='whether to sleep to lower the fps')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--sound-threshold',
        type=float,
        default=0.6,
        help='recognition score threshold that activate sound reflection')
    parser.add_argument(
        '--average-size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    parser.add_argument(
        '--sample-interval',
        type=int,
        default=3,
        help='every sample_interval frames to sample one')
    parser.add_argument(
        '--drawing-fps',
        type=int,
        default=20,
        help='Set upper bound FPS value of the output drawing')
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=4,
        help='Set upper bound FPS value of model inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--sound-interval',
        type=int,
        default=3,
        help='set the least interval between two sound')

    # renyu: from mmpose
    parser.add_argument(
    '--show',
    action='store_true',
    default=False,
    help='whether to show img')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    args = parser.parse_args()
    assert args.drawing_fps >= 0 and args.inference_fps >= 0, \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args

def add_meme_image_on_origin_image(origin_img, action_name):
    if action_name in PICTURE_DICT:
        try:
            meme_overlay = cv2.imread(PICTURE_DICT[action_name])
        except:
            print("no picture ", PICTURE_DICT[action_name])
            return
       
        # renyu: constrain the max side length
        max_side_length = 400

        # renyu: keep the meme the fixed size, or it may overflow
        ov_height, ov_width = meme_overlay.shape[:2]

        if ov_height > ov_width:
            scale = max_side_length / ov_height
        else:
            scale = max_side_length / ov_width

        new_height = int(ov_height * scale)
        new_width = int(ov_width * scale)

        meme_overlay = cv2.resize(meme_overlay, (new_width, new_height), interpolation=cv2.INTER_AREA)

        x_offset = (CAMERAWIDTH - new_width) // 2
        y_offset = (CAMERAHEIGHT - new_height) // 2

        origin_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = meme_overlay

# renyu: human detection and pose estimation
def draw_skeleton_and_show_one_frame(args,
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
    global stars_number
    stars_number = len(bboxes) # renyu: update the number of people

    # renyu: show the human number in the lower left corner of screen
    text = 'Number of stars: ' + str(stars_number)
    location = (10, CAMERAHEIGHT - 60)
    cv2.putText(img, text, location, FONTFACE, FONTSCALE,
                                    FONTCOLOR, THICKNESS, LINETYPE)

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # renyu: show the processed image on the screen using visualizer (we don't need to cv2.imshow())
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
    # renyu: actually we don't need to return the result, it's copied from the demo to save an output file
    return data_samples.get('pred_instances', None)

# renyu: read one image from camera
#        then add action name from the action recognition thread (by queue)
#        then do the pose estimation to get the skeleton
#        show the processed image
def show_skeleton_and_action_from_camera():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    cur_time = time.time()
    
    camera_fps = camera.get(cv2.CAP_PROP_FPS)
    # renyu: default 640*480, we could reset the resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERAWIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERAHEIGHT)
    camera_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    camera_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('camera_fps:', camera_fps, ' camera_width:', camera_width, ' camera_height:', camera_height)
    
    count = 0
    sample_count = 0
    while True:
        no_action_msg = 'Waiting for action ...'
        _, frame = camera.read()

        # renyu: 640*480 is too small, fix expand it by resizing
        #        we could adjust it according to computer performance
        #        (deprecated: just adjust the camera resolution)
        #frame = cv2.resize(frame, (800, 600), interpolation= cv2.INTER_LINEAR)
        
        # renyu: bad computer needs to lower the fps
        if args.bad_computer is True:
            count += 1
            if (count % camera_fps != 0):
                continue
            count = 0

        # renyu: do frame sample here, but in the start phrase we don't wait
        sample_count += 1
        if len(frame_queue) < sample_length:
            frame_queue.append(np.array(frame[:, :, ::-1]))
        elif sample_count % sample_interval == 0:
            frame_queue.append(np.array(frame[:, :, ::-1]))
            sample_count = 0

        # renyu: read action recognition results from the queue and draw it on the frame
        if len(action_queue) != 0:
            text_info = {}
            results = action_queue.popleft()
            #print(results) # renyu: for debug, check the action recognition result
            for i, result in enumerate(results):
                selected_label, score = result
                # renyu: actions not meet the threshold, ignore (should not be, bacause filter in action recogition)
                if score < threshold:
                    break
                else:
                    location = (10, 40 + i * 20)
                    font_color = FONTCOLOR
                    
                    # renyu: actions not meet the sound threshold, normally show
                    #        actions meet the sound threshold, specially show
                    if score < sound_threshold:
                        text = selected_label + ': ' + str(round(score * 100, 2))
                    else:
                        text = selected_label + ': ' + str(round(score * 100, 2)) + ' !'
                        font_color = SPECIALFONTCOLOR
                        add_meme_image_on_origin_image(frame, selected_label)
                    
                    text_info[location] = text
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                    font_color, THICKNESS, LINETYPE)

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, no_action_msg, (10, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        # topdown pose estimation
        pred_instances = draw_skeleton_and_show_one_frame(args, frame, detector,
                                                          pose_estimator, visualizer,
                                                          0.001)
        
        # renyu: if press 'ESC' 'q' 'Q' button, terminate the app
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            camera.release()
            cv2.destroyAllWindows()
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

# renyu: do the action recognition task and put the results into the queue (to show_result thread and make_sound thread)
#        it needs multiple frames as input, we do it in another thread so it would be detached from the skeleton show
def action_recognize():
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    count = 0
    # renyu: bad computer waits for camera start
    time.sleep(10)
    print('action recognition starts.')
    while True:
        cur_windows = []

        # renyu: bad computer needs to lower the inference frequency
        if args.bad_computer is True:
            time.sleep(2)

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = pseudo_collate([cur_data])

        # Forward the model
        with torch.no_grad():
            result = model.test_step(cur_data)[0]
            #result = inference_recognizer(model, cur_data, test_pipeline)
        scores = result.pred_score.tolist()
        scores = np.array(scores)
        score_cache.append(scores)
        scores_sum += scores

        # renyu: for inference speed checking
        #count += 1
        #print("inference_count", count)

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            score_tuples = tuple(zip(label, scores_avg))
            score_sorted = sorted(
                score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:num_selected_labels]

            # renyu: add a filter, if the action label start with '#', drop it
            filtered_results = []
            for i in range(len(results)):
                action_name, action_score = results[i]
                if action_name.startswith("#"):
                    continue
                else:
                    filtered_results.append(results[i])

            # renyu: enqueue the action recognition result to show and make sound
            if len(filtered_results) > 0:
                action_queue.append(filtered_results)
                sound_queue.append(filtered_results[0])
            
            scores_sum -= score_cache.popleft()

            if inference_fps > 0:
                # add a limiter for actual inference fps <= inference_fps
                sleep_time = 1 / inference_fps - (time.time() - cur_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                cur_time = time.time()

# renyu: make sound according to the action type
def make_sound():
    cur_time = time.time()
    count = 0
    # renyu: bad computer waits for camera start
    time.sleep(10)
    print('Sound reflection starts.')
    while True:
        # renyu: avoid two sound too close
        time.sleep(args.sound_interval)

        if len(sound_queue) != 0:
                    # renyu: make sound according to the last action
                    #        check if it in the dict and meet the sound threshold
                    action_name, action_score = sound_queue.popleft()
                    if action_name in SOUND_DICT and action_score >= sound_threshold:
                        print("action: ", action_name, " play sound: ", SOUND_DICT[action_name])
                        #winsound.PlaySound(SOUND_DICT[action_name], winsound.SND_FILENAME)

                        # renyu: play sound and wait for its end
                        sound_effect = pygame.mixer.Sound(SOUND_DICT[action_name])
                        sound_effect.play()
                        time.sleep(sound_effect.get_length())
                    else:
                        print(action_name, "not in the list or not meet the threshold.")

def make_bgm():
    time.sleep(5)
    print('BGM starts.')
    while True:
        # renyu: avoid two sound too close
        time.sleep(args.sound_interval)

        global stars_number
        cur_people = stars_number
        if cur_people > 3:
            cur_people = 3
        bgm_name = random.sample(BGM_DICT[cur_people], 1)[0]
        print("stars number: ", cur_people, " play bgm: ", bgm_name)
        sound_bgm = pygame.mixer.Sound(bgm_name)
        bgm_channel = sound_bgm.play()

        # renyu: check if we need to change the music every 30s
        #        if one song is over or the people number changes, then change the music
        while True:
            time.sleep(30)
            if bgm_channel.get_busy() is not True or cur_people != stars_number:
                bgm_channel.fadeout(3000)
                time.sleep(3)
                break

def main():
    # renyu: make a lot of global value to simplify the parameter pass between threads
    global average_size, threshold, sound_threshold, drawing_fps, inference_fps, \
        device, model, camera, data, label, sample_length, \
        test_pipeline, frame_queue, action_queue, sound_queue, \
        detector, pose_estimator, visualizer, args, \
        stars_number, sample_interval

    args = parse_args()

    # renyu: bad_computer needs to lower the fps
    if args.bad_computer is True:
        args.average_size = 1
        args.inference_fps = 1

    average_size = args.average_size
    threshold = args.threshold
    sound_threshold = args.sound_threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps
    sample_interval = args.sample_interval

    device = torch.device(args.device)

    # renyu: build human detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    stars_number = 0

    # renyu: build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # renyu: build visualizer (show the processed image with action and skeleton)
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    cfg = Config.fromfile(args.action_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # renyu: build action recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.action_checkpoint, device=args.device)
    camera = cv2.VideoCapture(args.camera_id)
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    # renyu: most openmmlab libs use local files as input, we have to change it to camera
    cfg = model.cfg
    sample_length = 0
    '''
    pipeline = None
    if cfg.test_pipeline is not None:
        print(cfg.test_pipeline)
        pipeline = cfg.test_pipeline
    elif cfg.val_pipeline is not None:
        print("no test pipeline config, try val pipeline.")
        pipeline = cfg.val_pipeline    # renyu: some config files don't have test_pipeline, use val instead
    else:
        print("no test or val pipeline config!")
        assert pipeline is not None
    '''

    # renyu: create the webcam real-time app pipeline
    #        an important change is that we can't sample too much frames for too long
    #        because we don't have a whole offline video and have enought time to process
    #        just remove the 'SampleFrames' step. instead, we sample every M frames and get N frames as input
    pipeline = cfg.test_pipeline
    print(pipeline)
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in get_str_type(step['type']):
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if get_str_type(step['type']) in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    # renyu: init the sound player
    pygame.mixer.init()

    try:
        frame_queue = deque(maxlen=sample_length)
        action_queue = deque(maxlen=1)
        sound_queue = deque(maxlen=1)
        
        thread_show = Thread(target=show_skeleton_and_action_from_camera, args=(), daemon=True)
        thread_recognize = Thread(target=action_recognize, args=(), daemon=True)
        thread_sound = Thread(target=make_sound, args=(), daemon=True)
        thread_bgm = Thread(target=make_bgm, args=(), daemon=True)
        
        thread_show.start()
        thread_recognize.start()
        thread_sound.start()
        thread_bgm.start()
        
        # renyu: if show thread quits then quit the whole app
        thread_show.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
