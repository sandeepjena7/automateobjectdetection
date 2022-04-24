import cv2
import torch
import numpy as np
from pathlib import Path
import tensorflow as tf
import time
import os
import json
import sys
sys.path.insert(0, 'src/yolov5')

from src.yolov5.models.common import DetectMultiBackend
from src.yolov5.utils.datasets import  LoadImages,IMG_FORMATS, VID_FORMATS
from src.yolov5.utils.general import ( check_img_size,  cv2,
                           non_max_suppression,  scale_coords, )
from src.yolov5.utils.torch_utils import select_device, time_sync
from src.yolov5.utils.plots import Annotator, colors

from src.detectron2.config import get_cfg
from src.detectron2.engine import DefaultPredictor

from src.utils.predfilter import detectron_filter,tf1_filter


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
    def __call__(self):
        
        since = time.time()
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



            for i ,det in enumerate(pred):
                count = len(det)
                xmin,ymin,xmax,ymax = [None]*count,[None]*count,[None]*count,[None]*count
                seen +=1

                p, im0,  = path, im0s.copy()

                p = Path(p)
                save_path = str(Path(self.save_dir) / p.name)
                annotator = Annotator(im0, line_width=1, example=str(names))

                if len(det):
                    
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()


                    for j,(*xyxy ,conf,cls) in enumerate( det):

                        if dataset.mode == 'image':
                            
                            xmin[j], ymin[j],xmax[j],ymax[j] =  int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])


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

        time_elapsed = time.time() - since
        time_send = f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        t = tuple(x / seen * 1E3 for x in dt)
        print(f'Speed per image :- ::: %.1fms pre-process ::: %.1fms inference ::: %.1fms custom filter ::: per image at reshape {(1, 3, imgsz,imgsz)}' % t)
        print(":::",f'Predcitition complete in {time_send} ',":::")
        
        value = {}
        value['mode'] = dataset.mode
        value['time'] = time_send
        value['path'] = save_path

        if dataset.mode == 'image':
            if len(xmin): 
                value['count'] = count
                value['xmin'],value['ymin'],value['xmax'],value['ymax'] = xmin,ymin,xmax,ymax
                return value

            else:
                value['xmin'],value['ymin'],value['xmax'],value['ymax'] = 0,0,0,0
                return value
        else:
            return value





class Detectron2:
    def __init__(self
                ,modelpath
                ,configymlpath
                ,inputpath
                ,save_dir
                ,output_json_path = None
                ,img_size = 416
                ,conf_thres = 0.5
                ,save_img = True
                ,hide_conf = False
                ,hide_labels = False
                ,view_img = False
                ):
        # bro dont use video for today i fix it tomorrow the send you code  found big bug
        
        # self.model = modelyamlpath
        self.cfg = get_cfg()
        self.cfg.merge_from_file(configymlpath)

        if output_json_path is not None:
            self.names = self._read_json(output_json_path)
        else:
            self.names = None

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
    
    def _read_json(self,detectron2trainjonpath):

        p = str(Path(detectron2trainjonpath).resolve()) 
        assert os.path.isfile(p) , 'detectron2 train.json path is not right'

        data = json.load(open(p,'r'))
        names = []

        for cat in data['categories']:
            names.append(cat['name'])

        return names

    def __call__(self):

        since = time.time()
        source = str(self.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

        assert is_file , "image/video file not found "

        predictor = DefaultPredictor(self.cfg)  # change source code
        imgsz = check_img_size(self.img_size)
        dataset = LoadImages(self.source,img_size=imgsz)
        bs = 1

        vid_path,vid_writer = [None]*bs,[None]*bs

        dt, seen = [0.0, 0.0, 0.0], 0
        

        for path,im,im0s,vid_cap,s in dataset:
            t1 = time_sync()

            img = im.transpose((1,2,0))
            pred,t2,t3 = predictor(img) # source some modfication

            dt[0] += t2 - t1 
            dt[1] += t3 - t2

            pred = detectron_filter(pred)
            dt[2] = time_sync() - t3


            for i,det in enumerate(pred):
                count = len(det)
                xmin,ymin,xmax,ymax = [None]*count,[None]*count,[None]*count,[None]*count
                seen +=1

                im0 = im0s.copy()

                p = Path(path)

                save_path = str(Path(self.save_dir) / p.name)

                annotator = Annotator(im0,line_width=1)

                if len(det):
                    det = np.array(det) # i have using slice the array
                    det[:, :4] = scale_coords(im.shape[1:], det[:, :4], im0.shape).round()

                    for j,(*xyxy ,conf,cls) in enumerate( det):

                        if dataset.mode == 'image':    
                            xmin[j], ymin[j],xmax[j],ymax[j] =  int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])


                        if self.save_img or self.view_img :
                            c = int(cls)
                            if self.names is not None:
                                label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf}')
                            else:
                                label = None if self.hide_labels else (c if self.hide_conf else f'{c} {conf}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                
                im0 = annotator.result()

                if self.view_img:
                    cv2.imshow(str(p),im0)
                    cv2.waitKey(1)
                
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path,im0)

                    else:  
                        if vid_path[i] != save_path:
                            vid_path[i] = save_path
                            
                            if isinstance(vid_writer[i],cv2.VideoWriter):
                                vid_writer[i].release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            save_path = str(Path(save_path).with_suffix('.mp4'))
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            
                        vid_writer[i].write(im0)


        time_elapsed = time.time() - since
        time_send = f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        t = tuple(x / seen * 1E3 for x in dt)
        print(f'Speed per image :- ::: %.1fms pre-process ::: %.1fms inference ::: %.1fms custom filter ::: per image at reshape {(1, 3, imgsz,imgsz)}' % t)
        print(":::",f'Predcitition complete in {time_send} ',":::")
        
        value = {}
        value['mode'] = dataset.mode
        value['time'] = time_send
        value['path'] = save_path

        if dataset.mode == 'image':
            if len(xmin): 
                value['count'] = count
                value['xmin'],value['ymin'],value['xmax'],value['ymax'] = xmin,ymin,xmax,ymax
                return value

            else:
                value['xmin'],value['ymin'],value['xmax'],value['ymax'] = 0,0,0,0
                return value
        else:
            return value



class TF1: 
    def __init__(self
                ,modelpath
                ,labelmappath
                ,inputpath
                ,save_dir
                ,img_size = 416
                ,conf_thres = 0.5
                ,save_img = True
                ,hide_conf = False
                ,hide_labels = False
                ,view_img = False):

        self.names = self._read_label_map(labelmappath) 
    
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
            with tf.gfile.GFile(modelpath, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
     
    def _read_label_map(self,label_map_path):
        # https://github.com/datitran/raccoon_dataset/pull/93/files
        # also added few line as per our requirements
        item_id = None
        item_name = None
        items = {}
        p = str(Path(label_map_path).resolve())
        assert os.path.isfile(p) , "labelmapt path is not right"

        with open(label_map_path, "r") as file:
            for line in file:
                line.replace(" ", "")
                if line == "item{":
                    pass
                elif line == "}":
                    pass
                elif "id" in line:
                    item_id = int(line.split(":", 1)[1].strip())
                elif "name" in line:
                    item_name = line.split(":", 1)[1].replace("'", "").strip().replace('"'," ")

                if item_id is not None and item_name is not None:
                    items[item_id] = item_name
                    item_id = None
                    item_name = None

        return items


    def __call__(self):

        since = time.time()
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

            t1 = time_sync()
            img = im.transpose((1,2,0))[:, :, ::-1]

            
            if len(img.shape) == 3:
                img = img[None] # expand dims
            t2 = time_sync()
            dt[0] += t2 - t1

            (boxes, scores, classes, num) = sess.run(
                [self.detection_boxes,self.detection_scores,self.detection_classes,self.num_detections],
                feed_dict = {self.image_tensor:img} )
            t3 = time_sync()
            dt[1] += t3 - t2

            pred = tf1_filter(im,boxes,scores,classes,self.conf_thres)
            dt[2] += time_sync() - t3


            for i,det in enumerate(pred):
                count = len(det)
                xmin,ymin,xmax,ymax = [None]*count,[None]*count,[None]*count,[None]*count
                seen += 1
            
                im0 = im0s.copy()
                p = Path(path)
                save_path = str(Path(self.save_dir) / p.name)
                annotator = Annotator(im0,line_width=1)

                if len(det):
                    det = np.array(det)
                    det[:, :4] = scale_coords(im.shape[1:], det[:, :4], im0.shape).round()
                    
                    for j,(*xyxy ,conf,cls) in enumerate( det):

                        if dataset.mode == 'image': 
                            xmin[j], ymin[j],xmax[j],ymax[j] =  int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                        if self.save_img or self.view_img:
                            c = int(cls)
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                im0 = annotator.result()

                if self.view_img:
                    cv2.imshow(str(p),im0)
                    cv2.waitKey(1)

                
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path,im0)

                    else:  
                        if vid_path[i] != save_path:
                            vid_path[i] = save_path
                            
                            if isinstance(vid_writer[i],cv2.VideoWriter):
                                vid_writer[i].release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            save_path = str(Path(save_path).with_suffix('.mp4'))
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            
                        vid_writer[i].write(im0)

        time_elapsed = time.time() - since
        time_send = f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        t = tuple(x / seen * 1E3 for x in dt)
        print(f'Speed per image :- ::: %.1fms pre-process ::: %.1fms inference ::: %.1fms custom filter ::: per image at reshape {(1, 3, imgsz,imgsz)}' % t)
        print(":::",f'Predcitition complete in {time_send} ',":::")
        
        value = {}
        value['mode'] = dataset.mode
        value['time'] = time_send
        value['path'] = save_path

        if dataset.mode == 'image':
            if len(xmin): 
                value['count'] = count
                value['xmin'],value['ymin'],value['xmax'],value['ymax'] = xmin,ymin,xmax,ymax
                return value

            else:
                value['xmin'],value['ymin'],value['xmax'],value['ymax'] = 0,0,0,0
                return value
        else:
            return value
