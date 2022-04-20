# Author: Sourangshu Pal
# Date: 15/11/19
# Import packages
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from src.tf1.utils import label_map_util
from src.tf1.utils import visualization_utils as vis_util



class CardsDetector:
    def __init__(self):
        # This is needed since the notebook is stored in the tf1 folder.
        sys.path.append("..")
        # Name of the directory containing the object detection module we're using
        # self.MODEL_NAME = modelPath
        # self.IMAGE_NAME = imagePath
        # # print(self.IMAGE_NAME)
        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        self.PATH_TO_CKPT ="frozen_inference_graph.pb"
        # Path to label map file
        self.PATH_TO_LABELS = "labelmap.pbtxt"
        # self.PATH_TO_LABELS = "data/labelmap.pbtxt"
        # Path to images
        self.PATH_TO_IMAGE = "inputImage.jpg"
        #print(self.PATH_TO_IMAGE)
        # Number of classes the object detector can identify
        self.NUM_CLASSES = 6

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.class_names_mapping = {
            1: "Nine", 2: "Ten",3: "jack", 4: "queen", 5: "King", 6:"Ace"
        }
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def getPrediction(self):

        # https://danijar.com/what-is-a-tensorflow-session/ very good desribed in tf sesssion
        # Load the Tensorflow model into memory.
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        sess = tf.Session(graph=self.detection_graph)
        image = cv2.imread(self.PATH_TO_IMAGE)

        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        result = scores.flatten()
        res = []
        for idx in range(0, len(result)):
            if result[idx] > .40:
                res.append(idx)

  
        dd = [idx for idx in range(0,scores.size) if scores[0,:][idx] > 0.40]
        resss = [idx for idx in range(0,len(result)) if result[idx] > .40]
        top_classes = classes.flatten()
        bbb = boxes[0,:]
        yxyx =  [boxes[0,:][idx]for idx in dd]
        for i in yxyx:
            ymin, xmin, ymax, xmax = i[0],i[1],i[2],i[3]

        ymin*image.shape[0]
        ymax*image.shape[0]
        xmin*image.shape[1]
        xmax*image.shape[1]

        # Selecting class 2 and 3
        # top_classes = top_classes[top_classes > 1]
        clasesssss = [classes[0,:][i] for i in res]
        res_list = [top_classes[i] for i in res]

        class_final_names = [self.class_names_mapping[x] for x in res_list]
        sss_scres = [scores[0,:][idx] for idx in res]
        top_scores = [e for l2 in scores for e in l2 if e > 0.30]
        # top_scores_san = [e for e in result if e > 0.30]
        # final_output = list(zip(class_final_names, top_scores))

        # print(final_output)

        # new_classes = classes.flatten()
        new_scores = scores.flatten()
        # ss = boxes.squeeze()

        new_boxes = boxes.reshape(300, 4)

        # get all boxes from an array
        max_boxes_to_draw = new_boxes.shape[0]
        max_bo =  boxes.shape[1]
        # this is set as a default but feel free to adjust it to your needs
        min_score_thresh = .30
        # iterate over all objects found
        # boundingBox = {}
        # for i in range(min(max_boxes_to_draw, new_boxes.shape[0])):
        #     if new_scores is None or new_scores[i] > min_score_thresh:
        #         boundingBox[class_final_names[i]] = new_boxes[i]
        #         print("Bounding Boxes of", class_final_names[i], new_boxes[i])

        listOfOutput = []
        for (name, score, i) in zip(class_final_names, top_scores, range(min(max_boxes_to_draw, new_boxes.shape[0]))):
            valDict = {}
            valDict["className"] = name
            valDict["confidence"] = str(score)
            if new_scores is None or new_scores[i] > min_score_thresh:
                val = list(new_boxes[i])
                valDict["yMin"] = str(val[0])
                valDict["xMin"] = str(val[1])
                valDict["yMax"] = str(val[2])
                valDict["xMax"] = str(val[3])
                listOfOutput.append(valDict)
        # new_boxes = boxes.reshape(100,4)
        # print(new_boxes)
        # print(type(new_boxes))
        # print(new_boxes.shape)
        # print(boxes.shape)
        # Draw the results of the detection (aka 'visulaize the results')

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        output_filename = 'output4.jpg'
        cv2.imwrite(output_filename, image)
        # opencodedbase64 = encodeImageIntoBase64("output4.jpg")
        # # json_image = dict(zip(img_dict, image_64_encode_list))
        # # print(open_output_image)
        # # plt.savefig(PATH + '\\' + arr.split('.')[0] + '_labeled.jpg')
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # #
        # # # All the results have been drawn on image. Now display the image.
        # # cv2.imshow('Object detector', image)
        # #
        # # # Press any key to close the image
        # # cv2.waitKey(0)
        # #
        # # # Clean up
        # # cv2.destroyAllWindows()
        # listOfOutput.append({"image": opencodedbase64.decode('utf-8')})
        return "cool"


# xmin * im_width, xmax * im_width,
# ymin * im_height, ymax * im_height

# ymin, xmin, ymax, xmax

# defaultdict(<class 'str'>, {(0.08064154535531998, 0.3132706582546234, 0.48428502678871155, 0.49634745717048645): 'Aqua', 
# (0.5224219560623169, 0.5123851299285889, 0.9805923104286194, 0.7070580720901489): 
# 'Aqua', (0.06725195050239563, 0.5148552060127258, 0.45828887820243835, 0.6988524794578552): 
# 'Aqua', (0.5208129286766052, 0.32283133268356323, 0.9629364013671875, 0.5039094090461731): 'Aqua'})