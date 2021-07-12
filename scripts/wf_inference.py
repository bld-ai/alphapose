"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort

from detector.apis import get_detector
from detector.yolo.darknet import Darknet
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

""" Run format
python scripts/wf_inference.py \
--cfg path/to/config \
--checkpoint path/to/pretrained_model \
--mode video \
--sp \
--indir path/to/videos/folder \
--profile \
--debug \
--detbatch 2 \
--posebatch 80 \
--gpus 0

"""

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--mode', dest='mode',
                    help='input mode, option: image/video', default="")
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='videos directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

# Initial Setup
args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

if not (args.mode == "image" or args.mode == "video"):
    raise IOError('Error: --mode must be one of the options: image/video')

if args.save_video:
    from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt


# def 


if __name__ == "__main__":
    # check if output path exists
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Get YOLO Model
    start_time = getTime()
    print('Loading YOLO model..')
    yolo_model = Darknet(cfg.get('CONFIG', 'detector/yolo/cfg/yolov3-spp.cfg'))
    yolo_model.load_weights(cfg.get('WEIGHTS', 'detector/yolo/data/yolov3-spp.weights'))
    yolo_model.net_info['height'] = cfg.get('INP_DIM', 608)
    if len(args.gpus) > 1:
        yolo_model = torch.nn.DataParallel(yolo_model, device_ids=args.gpus).to(args.device)
    else:
        yolo_model.to(args.device)
    yolo_model.eval()
    ckpt, load_yolo_time = getTime(start_time)
    print(f"Loading YOLO model finished in {load_yolo_time} seconds.")

    # Load Pose Model (Loading pose model)
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()
    _, load_pose_time = getTime(ckpt)
    print(f"Loading YOLO model finished in {load_pose_time} seconds.")


    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }


    if args.mode == "video" and len(args.inputpath):
        # search for mp4 videos, add video filepaths with .mp4 extensions to list
        for _, _, vid_files in os.walk(args.inputpath):
            break
        for videofile in vid_files:
            # load detection loader
            input_source = os.path.join(args.inputpath, videofile)
            filename = os.path.splitext(os.path.basename(input_source))[0]
            if args.debug:
                print(f"Processing file " + videofile)


            queueSize = args.qsize
            batchSize = args.posebatch
            if args.flip:
                batchSize = int(batchSize / 2)

            # Init data writer
            if args.save_video:
                video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + videofile)
                video_save_opt.update(det_loader.videoinfo)
                writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize, filename=filename).start()
            else:
                writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize, filename=filename).start()

            det_loader = DetectionLoader(input_source, get_detector(args, model=yolo_model), cfg, args, batchSize=args.detbatch, mode=args.mode, queueSize=args.qsize)
            det_worker = det_loader.start()
            data_len = det_loader.length

            im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
            
            try:
                for i in im_names_desc:
                    start_time = getTime()
                    with torch.no_grad():
                        (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                        if orig_img is None:
                            break
                        if boxes is None or boxes.nelement() == 0:
                            writer.save(None, None, None, None, None, orig_img, im_name)
                            continue
                        if args.profile:
                            ckpt_time, det_time = getTime(start_time)
                            runtime_profile['dt'].append(det_time)
                        # Pose Estimation
                        inps = inps.to(args.device)
                        datalen = inps.size(0)
                        leftover = 0
                        if (datalen) % batchSize:
                            leftover = 1
                        num_batches = datalen // batchSize + leftover
                        hm = []
                        for j in range(num_batches):
                            inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                            if args.flip:
                                inps_j = torch.cat((inps_j, flip(inps_j)))
                            hm_j = pose_model(inps_j)
                            if args.flip:
                                hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                                hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                            hm.append(hm_j)
                        hm = torch.cat(hm)
                        if args.profile:
                            ckpt_time, pose_time = getTime(ckpt_time)
                            runtime_profile['pt'].append(pose_time)
                        hm = hm.cpu()
                        writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                        if args.profile:
                            ckpt_time, post_time = getTime(ckpt_time)
                            runtime_profile['pn'].append(post_time)
                    if args.profile:
                        #TQDM
                        im_names_desc.set_description(
                            'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                        )
                while(writer.running()):
                    time.sleep(1)
                    print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
                writer.stop()
                det_loader.stop()
            except Exception as e:
                print(repr(e))
                print('An error occurs when processing the images, please check it')
                pass
            except KeyboardInterrupt:
                # Thread won't be killed when press Ctrl+C
                if args.sp:
                    det_loader.terminate()
                    while(writer.running()):
                        time.sleep(1)
                        print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
                    writer.stop()
                else:
                    # subprocesses are killed, manually clear queues

                    det_loader.terminate()
                    writer.terminate()
                    writer.clear_queues()
                    det_loader.clear_queues()
    else:
        # normal processing of images
        pass