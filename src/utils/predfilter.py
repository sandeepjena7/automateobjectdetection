

def detectron_filter(pred):

    predictions = pred["instances"].to("cpu")
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    bbboxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    output = []

    if None not in (scores,classes,bbboxes):
        
        for box ,scor,cls in zip(bbboxes.tensor,scores,classes):
            oneplace = []
            for bb in box:
                oneplace.append(float(bb))
            oneplace.append(round(float(scor),2))
            oneplace.append(int(cls))
            output.append(oneplace)
    
    return [output]



def tf1_filter(img,boxes,scores,classes,conf_thres):

    idxs = [idx for idx in range(0,scores.size) if scores[0,:][idx] > conf_thres]
    clas = [classes[0,:][idx] for idx in idxs]
    score = [scores[0,:][idx] for idx in idxs]
    yxyx_NN = [boxes[0,:][idx]for idx in idxs]
    
    output = []

    width_img,height_img = img.shape[2],img.shape[1]
    if len(idxs):
        for yxyx,scor,cls in zip(yxyx_NN,score,clas):
            ymin, xmin, ymax, xmax = yxyx[0],yxyx[1],yxyx[2],yxyx[3]
                        
            oneplace = [ float(xmin*width_img)
                        ,float(ymin*height_img)
                        ,float(xmax*width_img)
                        ,float(ymax*height_img)
                        ,round(float(scor),2)
                        ,int(cls)
                        ]
            output.append(oneplace)
            
    return [output]
