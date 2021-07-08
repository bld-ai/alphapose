Runs
====

## Basic, 4s clip, 117 frames
It seems slow when loading the model for the first time.
CPU 100%, GPU 100%
```
 ~#@❯  python scripts/demo_inference.py --profile --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/vids/funk-4s.mp4
Loading pose model from pretrained_models/fast_res50_256x192.pth...
Loading YOLO model..
det time: 0.0512 | pose time: 0.0984 | post processing: 0.0586: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:24<00:00,  4.77it/s]
det time: 5.9850 | pose time: 11.5079 | post processing: 6.8508
===========================> Finish Model Running.
Results have been written to json.

 ~#@❯  python scripts/demo_inference.py --profile --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/vids/funk-4s.mp4
Loading pose model from pretrained_models/fast_res50_256x192.pth...
Loading YOLO model..
det time: 0.0188 | pose time: 0.0517 | post processing: 0.0619: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:15<00:00,  7.44it/s]
det time: 2.2037 | pose time: 6.0525 | post processing: 7.2376
===========================> Finish Model Running.

 ~#@❯  python scripts/demo_inference.py --profile --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/vids/funk-4s.mp4
Loading pose model from pretrained_models/fast_res50_256x192.pth...
Loading YOLO model..
det time: 0.0180 | pose time: 0.0525 | post processing: 0.0509: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:14<00:00,  8.12it/s]
det time: 2.1084 | pose time: 6.1419 | post processing: 5.9547
===========================> Finish Model Running.
Results have been written to json.

 ~#@❯  python scripts/demo_inference.py --profile --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/vids/funk-4s.mp4 
Loading pose model from pretrained_models/fast_res50_256x192.pth...
Loading YOLO model..
det time: 0.0136 | pose time: 0.0469 | post processing: 0.0671: 100%|███████████████████████████████████████████████████████████████████| 117/117 [00:15<00:00,  7.73it/s]
det time: 1.5856 | pose time: 5.4839 | post processing: 7.8489 |=> total: 14.92
===========================> Finish Model Running.
===========================> Rendering remaining 0 images in the queue...
Results have been written to json.

 ~#@❯  python scripts/demo_inference.py --profile --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/vids/funk-4s.mp4 
Loading pose model from pretrained_models/fast_res50_256x192.pth...
Loading YOLO model..
det time: 0.0136 | pose time: 0.0518 | post processing: 0.0696: 100%|███████████████████████████████████████████████████████████████████| 117/117 [00:15<00:00,  7.32it/s]
det time: 1.5905 | pose time: 6.0639 | post processing: 8.1387 |=> total: 15.79
===========================> Finish Model Running.
```

## Basic, 10s clip, 182 frames
CPU 100%, GPU 100%
```
 ~#@❯  python scripts/demo_inference.py --profile --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/vids/video-18s-10fps.mp4 
Loading pose model from pretrained_models/fast_res50_256x192.pth...
Loading YOLO model..
det time: 0.0222 | pose time: 0.0577 | post processing: 0.0214: 100%|███████████████████████████████████████████████████████████████████| 182/182 [00:18<00:00,  9.74it/s]
det time: 4.0387 | pose time: 10.4986 | post processing: 3.8870 |=> total: 18.42
===========================> Finish Model Running.

Results have been written to json.
 ~#@❯  python scripts/demo_inference.py --profile --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/vids/video-18s-10fps.mp4 
Loading pose model from pretrained_models/fast_res50_256x192.pth...
Loading YOLO model..
det time: 0.0239 | pose time: 0.0602 | post processing: 0.0150: 100%|███████████████████████████████████████████████████████████████████| 182/182 [00:18<00:00,  9.93it/s]
det time: 4.3485 | pose time: 10.9501 | post processing: 2.7266 |=> total: 18.03
===========================> Finish Model Running.
Results have been written to json.
```

## Basic 20s clip, 1288 frames
Stuck, full GPU memory, no utilization
Waiting for some process to finish

## Basic 60s clip, 1798 frames
Stuck, full GPU memory, no utilization