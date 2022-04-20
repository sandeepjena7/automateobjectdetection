import cv2 as cv
import json
from src.detectron2.engine import DefaultPredictor
from src.detectron2.config import get_cfg
from src.detectron2.utils.visualizer import Visualizer
from src.detectron2.utils.visualizer import ColorMode
from src.detectron2 import model_zoo
from src.detectron2.data import MetadataCatalog, DatasetCatalog
from src.detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image



class Detector:

	def __init__(self,filename):

		# set model and test set
		self.model = 'faster_rcnn_R_50_FPN_3x.yaml'
		self.filename = filename

		# obtain detectron2's default config
		self.cfg = get_cfg() 

		# load values from a file
		self.cfg.merge_from_file("config.yml")
		#self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model))

		# set device to cpu
		self.cfg.MODEL.DEVICE = "cpu"

		# get weights 
		# self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model) 
		#self.cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
		self.cfg.MODEL.WEIGHTS = "model_final.pth"

		# set the testing threshold for this model
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50

		# build model from weights
		# self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

	# build model and convert for inference
	def convert_model_for_inference(self):
		self.catgore_ = {
            1: "Nine", 2: "Ten",3: "jack", 4: "queen", 5: "King", 6:"Ace"
        }
		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		torch.save(model.state_dict(), 'checkpoint.pth')

		# return path to inference model
		return 'checkpoint.pth'


	def inference(self, file):

		predictor = DefaultPredictor(self.cfg)
		im = cv.imread(file)
		outputs = predictor(im)
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
		predictions = outputs["instances"].to("cpu")
		scores = predictions.scores if predictions.has("scores") else None
		classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
		boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
		for box in boxes.tensor:
			print(box)
		# visualise
		v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		predicted_image = v.get_image()
		im_rgb = cv.cvtColor(predicted_image, cv.COLOR_RGB2BGR)
		cv.imwrite('color_img.jpg', im_rgb)
		# imagekeeper = []
		
		return 'okkk'




