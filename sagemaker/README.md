# Documentation

## Notebook instance
* The benchmark runs were done on this [notebook instance](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/notebook-instances/efs-mount-notebook) or `arn:aws:sagemaker:us-east-2:204031725010:notebook-instance/efs-mount-notebook`.
* The prerequisite steps for installing alphapose are saved in [./notebook_instance.sh](./notebook_instance.sh).

## Script processor
* A `ScriptProcessor` was set up to run the custom `wf_inference.py`. This allows for handling multiple video files while loading a single model at the start of the run.
* The container is defined in [../script-processor.dockerfile](../script-processor.dockerfile) and requires [./setup.sh](./setup.sh).

### Run a processing job
```sh
python scripts/script_processor.py --s3_input s3://wf-sagemaker-us-east-2/inputvids/ --s3_output s3://wf-sagemaker-us-east-2/outputvids/
```

### Run wf_inference locally
This is an extension of demo_inference.py with an additional switch to accept a folder containing video files.
```sh
python scripts/wf_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --mode video
```