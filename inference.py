from utils import * 
import random
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
import torch, torchvision

logger = logging.getLogger("detectron2")

def config(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)

	# Set score_threshold for builtin models
	# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
	# cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45
	cfg.TEST.DETECTIONS_PER_IMAGE = 1000
	cfg.MODEL.WEIGHTS = args.weight
	cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000  # originally 1000
	cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 5000  # originally 1000
	cfg.freeze()
	return cfg

def get_parser():
	parser = argparse.ArgumentParser(description="Detectron2 inference for road stress")
	parser.add_argument("--config-file", required=True, metavar="FILE", help="path to config file")
	parser.add_argument("--dataset", required=True, help="path to dataset folder")
	parser.add_argument("--weight", required=True, metavar="FILE", help="path to weight file")
	parser.add_argument("--output", help="A file or directory to save output visualizations. If not given, will show output in an OpenCV window.")
	parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score for instance predictions to be shown")
	parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER,)
	return parser

def run_on_image(predictor, image):
	predictions = predictor(image)
	vis = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get("roadstress_train"), instance_mode=ColorMode.IMAGE)
	#vis = Visualizer(image[:, :, ::-1],
        #	metadata=MetadataCatalog.get("roadstress_train"), 
        #	scale=1.0, 
        #	instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
	#)
	if "instances" in predictions:
		instances = predictions["instances"].to(torch.device("cpu"))
		print(len(instances))
		# for i in range(0, 4):
		#	j = random.randint(0, len(instances) - 1)
		#	print(instances.get("pred_masks")[j])
		#	torchvision.utils.save_image(instances.get("pred_masks")[j], os.path.join(args.output, "instances_" + str(j) +".JPG"))
		#	cv2.imwrite(os.path.join(args.output, "instances_" + str(j) +".JPG"), instances.get("pred_masks")[j].numpy())
		for i in range(0, 4):
			j = random.randint(0, len(instances) - 1)
			vis.draw_sem_seg(instances.get("pred_masks")[j]).save(os.path.join(args.output, "instances_" + str(j) +".JPG"))
		vis_output = vis.draw_instance_predictions(predictions=instances)
		return predictions, vis_output
	else:
		print("Something wrong. Please check the inference.py script")

if __name__ == "__main__":
	args = get_parser().parse_args()
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	# Register the dataset:
	for d in ["train", "val"]:
		DatasetCatalog.register("roadstress_" + d, lambda d=d: get_roadstress_dicts("dataset/roadstress_new/" + d))
		MetadataCatalog.get("roadstress_" + d).set(thing_classes=["roadstress"])
		MetadataCatalog.get("roadstress_" + d).set(evaluator_type="coco")
		roadstress_metadata = MetadataCatalog.get("roadstress_train")
	print("Done Registering the dataset")


	cfg = config(args)
	print(cfg.dump())
	print("Finish Setting Up Config")

	predictor = DefaultPredictor(cfg)
	print("Start Inferencing")
	for img_path in glob.iglob(args.dataset + "/*.JPG"):
		#print(img_path)
		#print(os.path.basename(img_path))
		img = cv2.imread(img_path)	
		start_time = time.time()
		predictions, vis_img = run_on_image(predictor, img)
		logger.info(
                	"{}: {} in {:.2f}s".format(
                    		img_path,
                    		"detected {} instances".format(len(predictions["instances"]))
                    		if "instances" in predictions
                    		else "finished",
                    		time.time() - start_time,
                	)
            	)
		#vis_img.save(os.path.join(args.output, os.path.basename(img_path)))
	#model = build_model(cfg)
	#DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
	#do_test(cfg, model)