import cv2
from pyparsing import line
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(0, 'src/yolov5')

from src.yolov5.models.common import DetectMultiBackend
from src.yolov5.utils.datasets import LoadStreams, LoadImages,IMG_FORMATS, VID_FORMATS
from src.yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from src.yolov5.utils.torch_utils import select_device, time_sync
from src.yolov5.utils.plots import Annotator, colors, save_one_box

from src.detectron2.config import get_cfg
from src.detectron2.engine import DefaultPredictor


import tensorflow as tf

from src.tf1.utils import label_map_util
from src.tf1.utils import visualization_utils as vis_util


class YOLO:
    def __init__(self
                ,modelpath
                ,inputpath
                ,save_dir
                ,img_size = 416
                ,augment = True
                ,agnostic_nms = True
                ,conf_thres = 0.5
                ,iou_thres = 0.5
                ,save_img = True
                ,hide_conf = False
                ,hide_labels = False
                ,view_img = False):

        self.weights = modelpath
        self.source = inputpath
        self.save_dir = save_dir
        self.img_size = int(img_size)
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.device = "cpu"
        self.save_img = save_img
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.view_img = view_img
    
    @torch.no_grad()
    def getpredicton(self):
        
        source = str(self.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

        assert is_file , "image/video file not found "

        device = select_device(self.device)

        model = DetectMultiBackend(self.weights)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.img_size, s=stride)

        dataset = LoadImages(self.source,img_size=imgsz,stride=stride,auto=pt)
        bs = 1

        vid_path,vid_writer = [None]*bs,[None]*bs

        dt, seen = [0.0, 0.0, 0.0], 0

        for path,im,im0s,vid_cap,s in dataset:
            t1 = time_sync()

            im = torch.from_numpy(im).to(device)
            im = im.float()
            im /= 255

            if len(im.shape) == 3: 
                im = im[None] # expand dims
            t2 = time_sync()

            dt[0] += t2 - t1

            pred = model(im,augment=self.augment)
            t3 = time_sync()
            dt[1] += t3 - t2

            pred = non_max_suppression(pred,self.conf_thres,self.iou_thres,agnostic=self.agnostic_nms)
            dt[2] += time_sync() - t3

            bbbox = []

            for i ,det in enumerate(pred):
                seen +=1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)
                save_path = str(Path(self.save_dir) / p.name)
                annotator = Annotator(im0, line_width=3, example=str(names))

                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()


                    for *xyxy, conf, cls in reversed(det):

                        if dataset.mode == 'image':
                            bbbox.append(xyxy)

                        if self.save_img or self.view_img:
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                
                im0 = annotator.result()

                if self.view_img:
                    cv2.imshow(str(p),im0)
                    cv2.waitKey(1)
                
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

        t = tuple(x / seen * 1E3 for x in dt)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, imgsz,imgsz)}' % t)

        return "KGF"


class Detectron2:
    def __init__(self
                ,modelpath="model_final.pth"
                ,modelyamlpath='faster_rcnn_R_50_FPN_3x.yaml'
                ,configymlpath="config.yml"
                ,inputpath="file.jpg"
                ,save_dir='workdir'
                ,img_size = 416
                ,conf_thres = 0.5
                ,save_img = True
                ,hide_conf = False
                ,hide_labels = False
                ,view_img = False
                ):
        # bro dont use video for today i fix it tomorrow the send you code  found big bug
        
        self.model = modelyamlpath
        self.cfg = get_cfg()
        self.cfg.merge_from_file(configymlpath)
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.WEIGHTS = modelpath
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(conf_thres)
        self.source = inputpath
        self.save_dir = save_dir
        self.img_size = int(img_size)
        self.save_img = save_img
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.view_img = view_img

    def getpredicton(self):
        source = str(self.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

        assert is_file , "image/video file not found "

        predictor = DefaultPredictor(self.cfg)  # change source code
        imgsz = check_img_size(self.img_size)
        dataset = LoadImages(self.source,img_size=imgsz)
        bs = 1

        names = ["Nine", "Ten","jack", "queen", "King", "Ace"]

        vid_path,vid_writer = [None]*bs,[None]*bs

        dt, seen = [0.0, 0.0, 0.0], 0

        for path,im,im0s,vid_cap,s in dataset:
            t1 = time_sync()
            seen +=1

            img = im.transpose((1,2,0))
            pred = predictor(img)
            t2 = time_sync()
            dt[0] += t2 - t1

            predictions = pred["instances"].to("cpu")

            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
            bbboxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

            im0 = im0s.copy()

            p = Path(path)

            save_path = str(Path(self.save_dir) / p.name)

            annotator = Annotator(im0,line_width=3)

            bbboxes = scale_coords(im.shape[1:],bbboxes.tensor,im0.shape).round()
            count = 0

            for *xyxy ,conf,cls in zip(bbboxes,scores,classes):
                xyxy = xyxy[0]  # fix this issuse bboxes.tensor see
                if dataset.mode == 'image':
                    count +=1

                if self.save_img or self.view_img :
                    c = int(cls)
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            im0 = annotator.result()

            if self.view_img:
                cv2.imshow(str(p),im0)
                cv2.waitKey(1)
            
            if self.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path,im0)

                # else:  I have found big bug so donot use bro
                #     if vid_path[0] != save_path:
                #         vid_path[0] = save_path
                        
                #         if isinstance(vid_writer[0],cv2.VideoWriter):
                #             vid_writer[0].release()
                #         if vid_cap:
                #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                #         save_path = str(Path(save_path).with_suffix('.mp4'))
                #         vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        
                #     vid_writer[0].write(im0)




class Tf1: # i have not test bro i have test it tommorrow bro
    def __init__(self
                ,modelpath
                ,labelmappath
                ,inputpath
                ,save_dir='workdir'
                ,img_size = 416
                ,conf_thres = 0.5
                ,save_img = True
                ,hide_conf = False
                ,hide_labels = False
                ,view_img = False):

        self.names = read_labelmap(Path(labelmappath)) 
    
        self.conf_thres = float(conf_thres)
        self.source = inputpath
        self.save_dir = save_dir
        self.img_size = int(img_size)
        self.save_img = save_img
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.view_img = view_img

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(Path(modelpath), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        

        @staticmethod
        def read_labelmap(path):
            # https://github.com/datitran/raccoon_dataset/pull/93/files
            # but i have some modification for our requriements
            names = []
            with open(Path(path), "r") as file:
                for line in file:
                    line.replace(" ", "")
                    if "name" in line:
                        item_name = line.split(":", 1)[1].replace("'", "").strip()
                        if item_name is not None: 
                            names.append(item_name)
            return names

    def getpredicton(self):

        source = str(self.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        assert is_file , "image/video file not found "

        imgsz = check_img_size(self.img_size)
        dataset = LoadImages(self.source,img_size=imgsz)
        bs = 1

        sess = tf.Session(graph=self.detection_graph)
        vid_path,vid_writer = [None]*bs,[None]*bs

        dt, seen = [0.0, 0.0, 0.0], 0

        for path,im,im0s,vid_cap,s in dataset:
            seen += 1
            t1 = time_sync()
            img = im.transpose((1,2,0))[:, :, ::-1]
            
            if len(img.shape) == 3:
                img = img[None] # expand dims
            

            (boxes, scores, classes, num) = sess.run(
                [self.detection_boxes,self.detection_scores,self.detection_classes,self.num_detections],
                feed_dict = {self.image_tensor:img} )

            idxs = [idx for idx in range(0,scores.size) if scores[0,:][idx] > self.conf_thres]
            clas = [classes[i] for i in idxs]
            scores = [scores[0,:][idx] for idx in idxs]
            yxyx_NN = [boxes[0,:][idx]for idx in idxs] # NN - Not Normalized

            xyxy_down = []

            for i in yxyx_NN:
                ymin, xmin, ymax, xmax = i[0],i[1],i[2],i[3]  # in debuging please see the shape width x and height y

                xyxy_ = [ymin*img.shape[2]
                        ,ymax*img.shape[2]
                        ,xmin*img.shape[3]
                        ,xmax*img.shape[3]]

                xyxy_down.append(xyxy_)
            
            im0 = im0s.copy()
            p = Path(path)
            save_path = str(Path(self.save_dir) / p.name)
            annotator = Annotator(im0,line_width=3)

            bbboxes = scale_coords(im.shape[1:],xyxy_down,im0.shape).round()

            for *xyxy ,conf,cls in zip(bbboxes,scores,clas):
                if self.save_img or self.view_img:
                    c = int(cls)
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

            if self.view_img:
                cv2.imshow(str(p),im0)
                cv2.waitKey(1)

            
            if self.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path,im0)

