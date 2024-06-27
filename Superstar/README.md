# run command
python webcam_skeleton_action.py rtmdet_nano_320-8xb32_coco-person.py rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth rtmpose-m_8xb256-420e_coco-256x192.py rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth tsn_r50_1x1x8_video_infer.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth label_map_k400.txt --average-size 5 --threshold 0.2 --show

# TODO
detector and pose estimator are really good.
But the action recognizer (tsn_r50_1x1x8) is too bad. Maybe we could try some other pretrained model from mmaction2 model zoo. https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html (My computer is too weak to run them without GPU...)
Using other pretrained model seems easy. Just download the .pth file and find the correspond config file (you can download it or find it in mmaction2/config/xxx). However, you may meet some problem with the config file. First, you need to get the config file and its "_base_" file in current directory. Second, you have to make sure the "test_pipeline" config is suitable. (May need to modify it refer to the current config file)

Get new action types and .wav audio files and write them in SOUND_DICT in 'webcam_skeleton_action.py'.