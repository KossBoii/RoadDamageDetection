from utils import *
import sys
logger = logging.getLogger("detectron2")

'''
    Backbone model:
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

        - "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" 

''' 
def get_config(args, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.backbone))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.backbone)    

    # dataset configuration  
    cfg.DATASETS.TRAIN = (args.training_dataset + "_train",)
    cfg.DATASETS.TEST = (args.training_dataset + "_train", args.training_dataset + "_val",)
    
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 2                    # 2 GPUs --> each GPU will see 1 image per batch
    cfg.SOLVER.WARMUP_ITERS = 2000                  # 
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.CHECKPOINT_PERIOD = 20000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,16,32,64,128]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes                                   # number of categories

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.7
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 800

    #cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    #cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
    #cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    #cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000
    
    # Setup the logging folder
    with open('model_trained.txt', 'r') as f:
        model_trained = int(f.read())
    f.close()

    output_path = os.path.join(os.getcwd(), 'output')
    # Check if `output` folder already existed or not. If not, create the folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print('Done creating folder `output`')
    else:
        print('Folder `output` already existed!')

    # Create the new output folder for current model
    cur_model_output_path = os.path.join('./output', 'model_' + str(model_trained+1))
    if not os.path.exists(cur_model_output_path):
        os.makedirs(cur_model_output_path, exist_ok=True)
        print('Done creating output folder `model_{}` for current model'.format(model_trained+1))
    else:
        print('Folder model_{} already existed. Please double-check the code & the directory again'.format(model_trained+1))
    cfg.OUTPUT_DIR = cur_model_output_path
    model_trained = model_trained + 1
    with open('model_trained.txt', 'w') as f:
        f.write(str(model_trained))
    f.close()

    # configs from user's arguments
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

class Trainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=customMapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def main(args):
    # Check number of trained models
    with open('model_trained.txt', 'w') as f:
        if not os.path.exists(os.path.join(os.getcwd(), 'output')):
            f.write("0")                    # 0 models have been trained
        else:
            f.write(str(len(os.listdir(os.path.join(os.getcwd(), 'output')))))
    f.close()

    # Pre-process the annotation files
    dataset_basepath = os.path.join('./dataset', args.training_dataset)
    anno_path = os.path.join(dataset_basepath, 'train', 'via_export_json.json')
    thing_classes = pre_process_annos(anno_path)

    # Register User's Dataset
    for d in ['train', 'val']:
        DatasetCatalog.register(args.training_dataset + '_' + d, lambda d=d: get_dataset_dicts(os.path.join(dataset_basepath, d)))
        MetadataCatalog.get(args.training_dataset + '_' + d).set(thing_classes = thing_classes)
        MetadataCatalog.get(args.training_dataset + '_' + d).set(evaluator_type = 'coco')
    print('Done Registering the dataset!')

    # Configuration Initialization & Setup
    cfg = get_config(args, len(thing_classes))
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Training Starts from here
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    
    print('Finish Training the model! Start Running Evaluation...')
    res = Trainer.test(cfg, trainer.model)
    if comm.is_main_process():
        verify_results(cfg, res)
    return res

def custom_default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--training-dataset", required=True, help="dataset name to train")
    parser.add_argument("--backbone", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="backbone model")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    args = custom_default_argument_parser().parse_args()
    print("Command Line Args:", args)
    torch.backends.cudnn.benchmark = True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
