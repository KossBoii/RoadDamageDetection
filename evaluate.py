from utils import *

def get_config(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.getcwd(), 'output', args.model_name, 'config.yaml'))
    
    # configs from user's arguments
    cfg.merge_from_list(args.opts)
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 inference for road stress")
    parser.add_argument('--model-name', required=True, help="model name to be evaluated")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score for instance predictions to be shown")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER,)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    # Get configuration from the given trained model's config.yaml file & command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.getcwd(), 'output', args.model_name, 'config.yaml'))
    cfg.merge_from_list(args.opts)                  # configs from user's arguments

    dataset_name = cfg.DATASETS.TEST[0][:-6]

    # Pre-process the annotation files
    dataset_basepath = os.path.join('./dataset', dataset_name)
    anno_path = os.path.join(dataset_basepath, 'train', 'via_export_json.json')
    thing_classes = pre_process_annos(anno_path)

    # Register User's Dataset
    for d in ['train', 'val']:
        DatasetCatalog.register(dataset_name + '_' + d, lambda d=d: get_dataset_dicts(os.path.join(dataset_basepath, d)))
        MetadataCatalog.get(dataset_name + '_' + d).set(thing_classes = thing_classes)
        MetadataCatalog.get(dataset_name + '_' + d).set(evaluator_type = 'coco')
    print('Done Registering the dataset!')
    
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.WEIGHTS = os.path.join(os.getcwd(), 'output', args.model_name, 'model_final.pth')

    eval_output_path = os.path.join(os.getcwd(), 'eval_output')
    if not os.path.exists(eval_output_path):
        os.makedirs(eval_output_path, exist_ok=True)
    
    os.makedirs(os.path.join(eval_output_path, dataset_name), exist_ok=True)
    os.makedirs(os.path.join(eval_output_path, dataset_name, 'train'), exist_ok=True)
    os.makedirs(os.path.join(eval_output_path, dataset_name, 'val'), exist_ok=True)

    # COCOEvaluator
    predictor = DefaultPredictor(cfg)

    for d in ['train', 'val']:
        eval_output_dir = os.path.join(os.getcwd(), 'eval_output', dataset_name, d)
        evaluator = COCOEvaluator(dataset_name + '_' + d, output_dir=eval_output_dir)
        val_loader = build_detection_test_loader(cfg, dataset_name + '_' + d)
        print(inference_on_dataset(predictor.model, val_loader, evaluator))