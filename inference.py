from utils import *
logger = logging.getLogger("detectron2")

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

    parser.add_argument("--training-dataset", required=True, help="dataset name to train")
    parser.add_argument("--config-file", required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--training-dataset", required=True, help="dataset name to train")
    parser.add_argument("--dataset", required=True, help="path to dataset folder")
    parser.add_argument("--weight", required=True, metavar="FILE", help="path to weight file")
    parser.add_argument("--output", help="A file or directory to save output visualizations. If not given, will show output in an OpenCV window.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score for instance predictions to be shown")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER,)
    return parser

def run_on_image(predictor, dataset_name, image):
    predictions = predictor(image)
    dataset_metadata = MetadataCatalog.get(dataset_name + '_train')
    vis = Visualizer(image[:, :, ::-1], metadata=dataset_metadata, instance_mode=ColorMode.IMAGE)
    if 'instances' in predictions:
        instances = predictions['instances'].to(torch.device('cpu'))
        vis_output = vis.draw_instance_predictions(predictions=instances)
        return predictions, vis_output
    else:
        print("Something wrong. Please check the inference.py script")

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

    # Create inference output: 
    inference_output_path = os.path.join(os.getcwd(), 'prediction')
    if not os.path.exists(inference_output_path):
        os.makedirs(inference_output_path, exist_ok=True)
    
    os.makedirs(os.path.join(inference_output_path, dataset_name), exist_ok=True)

    predictor = DefaultPredictor(cfg)

    print('Start Inferencing')
    for file_name in os.listdir(args.img_path):
        temp = file_name.lower()
        if temp.endswith('.jpg') or temp.endswith('.jpeg') or temp.endswith('png'):
            # The file is an image
            img = cv2.imread(file_name)
            
            start_time = time.time()
            predictions, vis_img = run_on_image(predictor, dataset_name, img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    	file_name,
                    	"detected {} instances".format(len(predictions["instances"]))
                    	if "instances" in predictions
                    	else "finished",
                    	time.time() - start_time,
                	)
            	)
        
        # save the image to the prediction directory
        vis_img.save(os.path.join(inference_output_path, os.path.basename(file_name)))