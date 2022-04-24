# tf1 all quatnization model not work

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'

model_extensions = ["tar.gz",".pkl",".pt"]


yolov5_model_name = ["yolov5l","yolov5m","yolov5n","yolov5s","yolov5x"]


detecron2_PATH_TO_URL_SUFFIX = {
    "COCO-Detection/faster_rcnn_R_50_C4_1x": "137257644/model_final_721ade.pkl",
    "COCO-Detection/faster_rcnn_R_50_DC5_1x": "137847829/model_final_51d356.pkl",
    "COCO-Detection/faster_rcnn_R_50_FPN_1x": "137257794/model_final_b275ba.pkl",
    "COCO-Detection/faster_rcnn_R_50_C4_3x": "137849393/model_final_f97cb7.pkl",
    "COCO-Detection/faster_rcnn_R_50_DC5_3x": "137849425/model_final_68d202.pkl",
    "COCO-Detection/faster_rcnn_R_50_FPN_3x": "137849458/model_final_280758.pkl",
    "COCO-Detection/faster_rcnn_R_101_C4_3x": "138204752/model_final_298dad.pkl",
    "COCO-Detection/faster_rcnn_R_101_DC5_3x": "138204841/model_final_3e0943.pkl",
    "COCO-Detection/faster_rcnn_R_101_FPN_3x": "137851257/model_final_f6e8b1.pkl",
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x": "139173657/model_final_68b088.pkl",
    # COCO Detection with RetinaNet
    "COCO-Detection/retinanet_R_50_FPN_1x": "190397773/model_final_bfca0b.pkl",
    "COCO-Detection/retinanet_R_50_FPN_3x": "190397829/model_final_5bd44e.pkl",
    "COCO-Detection/retinanet_R_101_FPN_3x": "190397697/model_final_971ab9.pkl",
    # COCO Detection with RPN and Fast R-CNN
    "COCO-Detection/rpn_R_50_C4_1x": "137258005/model_final_450694.pkl",
    "COCO-Detection/rpn_R_50_FPN_1x": "137258492/model_final_02ce48.pkl",
    "COCO-Detection/fast_rcnn_R_50_FPN_1x": "137635226/model_final_e5f7ce.pkl",}


Tf1_PATH_TO_URL_SUFFIX = {
    "ssd_mobilenet_v1_coco": "ssd_mobilenet_v1_coco_2018_01_28.tar.gz"
    ,"ssd_mobilenet_v1_0.75_depth_coco": "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz"
    ,"ssd_mobilenet_v1_ppn_coco": "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz"
    ,"ssd_mobilenet_v1_fpn_coco": "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz"
    ,"ssd_resnet_50_fpn_coco": "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz"
    ,"ssd_mobilenet_v2_coco": "ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
    ,"ssdlite_mobilenet_v2_coco": "ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz"
    ,"ssd_inception_v2_coco": "ssd_inception_v2_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_inception_v2_coco":"faster_rcnn_inception_v2_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_resnet50_coco": "faster_rcnn_resnet50_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_resnet50_lowproposals_coco": "faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz"
    ,"rfcn_resnet101_coco": "rfcn_resnet101_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_resnet101_coco": "faster_rcnn_resnet101_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_resnet101_lowproposals_coco": "faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_inception_resnet_v2_atrous_coco": "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco": "faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_nas": "faster_rcnn_nas_coco_2018_01_28.tar.gz"
    ,"faster_rcnn_nas_lowproposals_coco": "faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz"
    ,"mask_rcnn_inception_resnet_v2_atrous_coco": "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz"
    ,"mask_rcnn_inception_v2_coco": "mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
    ,"mask_rcnn_resnet101_atrous_coco": "mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz"
    ,"mask_rcnn_resnet50_atrous_coco": "mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz"}




