from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
import argparse

parser = argparse.ArgumentParser(description='SageMaker Script Processor')
parser.add_argument('--image_uri', dest='image_uri',
                    help='URI of docker image', default='204031725010.dkr.ecr.us-east-2.amazonaws.com/bld-alphapose:v7')
parser.add_argument('--arn_role', dest='arn_role',
                    help='ARN role', default='arn:aws:iam::204031725010:role/wf-sagemaker-pose-pipeline')
parser.add_argument('--s3_input', dest='s3_input',
                    help='path to input in s3', default="s3://wf-sagemaker-us-east-2/inputvids/")
parser.add_argument('--s3_output', dest='s3_output',
                    help='path to output in s3', default="s3://wf-sagemaker-us-east-2/outputvids/")
parser.add_argument('--detbatch', dest='detbatch',
                    help='detector batch', default="2")

args = parser.parse_args()

script_processor = ScriptProcessor(
    command=['python'],
    image_uri=args.image_uri,
    role=args.arn_role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    max_runtime_in_seconds=5*60,
)

script_processor.run(
    code='scripts/wf_inference.py',
    inputs=[ProcessingInput(
        source=args.s3_input,
        destination='/opt/ml/processing/input'
    )],
    outputs=[ProcessingOutput(
        source='/opt/ml/processing/output/',
        destination=args.s3_output,
    )],
    arguments=[
        "--cfg", "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml", 
        "--checkpoint", "pretrained_models/fast_res50_256x192.pth", 
        "--mode", "video", 
        "--indir", "/opt/ml/processing/input/", 
        "--outdir", "/opt/ml/processing/output/", 
        "--detbatch", args.detbatch,
        "--sp",
        "--profile",
    ]
)
