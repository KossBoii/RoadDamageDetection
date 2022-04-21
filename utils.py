import argparse
import glob
import os
import cv2
import torch
import time
import logging
import os
import json
import copy
import numpy as np
import random
from datetime import datetime
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel

# Detectron2 Packages
from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.config import get_cfg, CfgNode
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

# Different Detectron2 Evaluators
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results
)

# Model Trainer Libraries
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

# Libraries for model logging
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

# Libraries for custom training loop
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
import re

logger = logging.getLogger("detectron2")

id_to_category = {}
category_to_id = {}

def pre_process_annos(path):
    thing_classes = []
    id = 0
    with open(path) as f:
        anno_dict = json.load(f)
        for key in anno_dict.keys():
            annos = anno_dict[key]['regions']
            for idx, anno in enumerate(annos):
                category = anno['region_attributes']['name']
                category = category.lower()
                category = re.sub(r"\s+", "", category, flags=re.UNICODE)

                if category not in category_to_id.keys():
                    category_to_id[category] = id
                    id_to_category[id] = category
                    id = id + 1

                    thing_classes.append(category)

                # fix annotations to be the same
                anno_dict[key]['regions'][idx]['region_attributes']['name'] = category
    
    # write the corrected annotation to the old annotation file
    with open(path, "w") as out:
        out.write(json.dumps(anno_dict, separators=(',', ':')))
    out.close()
    return thing_classes

def get_dataset_dicts(img_path):
    # Load and read json file stores information about annotations
    anno_path = os.path.join(img_path, 'via_export_json.json')
    if os.path.exists(anno_path):
        with open(anno_path) as f:
            anno_dict = json.load(f)

        dataset_dicts = []
        for idx, v in enumerate(anno_dict.values()):
            if(v['regions']):
                record = {}

                # open the image to get the height and width
                img_name = os.path.join(img_path, v['filename'])
                height, width  = cv2.imread(img_name).shape[:2]

                record['file_name'] = img_name
                record['image_id'] = idx
                record['height'] = height
                record['width'] = width

                # parse annotation for every instances in the image to record
                annos = v['regions']
                objs = []
                for anno in annos:
                    shape_attr = anno['shape_attributes']
                    class_name = anno['region_attributes']['name']
                    px = shape_attr['all_points_x']
                    py = shape_attr['all_points_y']

                    poly = [(x+0.5, y+0.5) for x,y in zip(px,py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        'bbox': [np.min(px), np.min(py), np.max(px), np.max(py)],
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'segmentation': [poly],
                        'category_id': category_to_id[class_name],
                        'is_crowd': 0
                    }
                    objs.append(obj)
                
                record['annotations'] = objs
                dataset_dicts.append(record)
    else:
        print('Please double-check to make sure that the annotation file named as via_export_json.json in %s' % anno_path)
    
    return dataset_dicts

'''
    Future Usage: Will try more image augmentation for model training
'''
def customMapper(dataset_dict):
  dataset_dict = copy.deepcopy(dataset_dict)
  image = utils.read_image(dataset_dict["file_name"], format="BGR")

  transform_list = [
                    T.Resize((600, 800)),
                    T.RandomFlip(prob=0.6, horizontal=True, vertical=False),
                    T.RandomFlip(prob=0.6, horizontal=False, vertical=True),
                    T.RandomBrightness(0.8, 1.3),
                    T.RandomSaturation(0.8, 1.3),
                    T.RandomContrast(0.8, 1.3)
                    ]
  image, transforms = T.apply_transform_gens(transform_list, image)
  dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
  annos = [
		utils.transform_instance_annotations(obj, transforms, image.shape[:2])
		for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0
    ]
  instances = utils.annotations_to_instances(annos, image.shape[:2])
  dataset_dict["instances"] = utils.filter_empty_instances(instances)
  return dataset_dict

# def get_evaluator(cfg, dataset_name, output_folder=None):
#     """
#     Create evaluator(s) for a given dataset.
#     This uses the special metadata "evaluator_type" associated with each builtin dataset.
#     For your own dataset, you can simply create an evaluator manually in your
#     script and do not have to worry about the hacky if-else logic here.
#     """
#     if output_folder is None:
#         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#     evaluator_list = []
#     evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#     if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
#         evaluator_list.append(
#             SemSegEvaluator(
#                 dataset_name,
#                 distributed=True,
#                 num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
#                 ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
#                 output_dir=output_folder,
#             )
#         )
#     if evaluator_type in ["coco", "coco_panoptic_seg"]:
#         evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
#     if evaluator_type == "coco_panoptic_seg":
#         evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
#     if evaluator_type == "cityscapes_instance":
#         assert (
#             torch.cuda.device_count() >= comm.get_rank()
#         ), "CityscapesEvaluator currently do not work with multiple machines."
#         return CityscapesInstanceEvaluator(dataset_name)
#     if evaluator_type == "cityscapes_sem_seg":
#         assert (
#             torch.cuda.device_count() >= comm.get_rank()
#         ), "CityscapesEvaluator currently do not work with multiple machines."
#         return CityscapesSemSegEvaluator(dataset_name)
#     if evaluator_type == "pascal_voc":
#         return PascalVOCDetectionEvaluator(dataset_name)
#     if evaluator_type == "lvis":
#         return LVISEvaluator(dataset_name, cfg, True, output_folder)
#     if len(evaluator_list) == 0:
#         raise NotImplementedError(
#             "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
#         )
#     if len(evaluator_list) == 1:
#         return evaluator_list[0]
#     return DatasetEvaluators(evaluator_list)




# '''
#     Evaluate the model performance with the COCO metrics (AP score)
# '''
# def do_test(cfg, model):
#     results = OrderedDict()
#     for dataset_name in cfg.DATASETS.TEST:          # perform inference on all testing dataset
#         data_loader = build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))
#         evaluator = get_evaluator(
#             cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
#         )
#         results_i = inference_on_dataset(model, data_loader, evaluator)
#         results[dataset_name] = results_i
#         if comm.is_main_process():
#             logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#             print_csv_format(results_i)
#     if len(results) == 1:
#         results = list(results.values())[0]
#     return results

# '''
#     Functionality: Custom training loop for detectron2
#     Usage: Will use to implement early stopping for model training
# '''
# def do_train(cfg, model, resume=False):
#     model.train()
#     optimizer = build_optimizer(cfg, model)
#     scheduler = build_lr_scheduler(cfg, optimizer)

#     checkpointer = DetectionCheckpointer(
#         model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
#     )
#     start_iter = (
#         checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
#     )
#     max_iter = cfg.SOLVER.MAX_ITER

#     periodic_checkpointer = PeriodicCheckpointer(
#         checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
#     )

#     writers = (
#         [
#             CommonMetricPrinter(max_iter),
#             JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
#             TensorboardXWriter(cfg.OUTPUT_DIR),
#         ]
#         if comm.is_main_process()
#         else []
#     )

#     # compared to "train_net.py", we do not support accurate timing and
#     # precise BN here, because they are not trivial to implement
#     data_loader = build_detection_train_loader(cfg, mapper=customMapper)
#     logger.info("Starting training from iteration {}".format(start_iter))
#     with EventStorage(start_iter) as storage:
#         for data, iteration in zip(data_loader, range(start_iter, max_iter)):
#             iteration = iteration + 1
#             storage.step()

#             loss_dict = model(data)
#             losses = sum(loss_dict.values())
#             assert torch.isfinite(losses).all(), loss_dict

#             loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
#             losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#             if comm.is_main_process():
#                 storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()
#             storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
#             scheduler.step()

#             if (
#                 cfg.TEST.EVAL_PERIOD > 0
#                 and iteration % cfg.TEST.EVAL_PERIOD == 0
#                 and iteration != max_iter
#             ):
#                 print("Run inference and evaluation inside do_train")
#                 do_test(cfg, model)
#                 comm.synchronize()

#             if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
#                 for writer in writers:
#                     writer.write()
#             periodic_checkpointer.step(iteration)