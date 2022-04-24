from src.utils.dirmakes import CreateDirectory,directorys,subdirectory,childdirectory
from src.utils.allutils import Tf1_PATH_TO_URL_SUFFIX,yolov5_model_name,detecron2_PATH_TO_URL_SUFFIX
from src.utils.downloadmodel import ModelUrl,modeldownload,Untar
from src.Prediction import YOLO,Detectron2,TF1


import os
from pathlib import Path
from colorama import Fore
import glob


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


class Framework(Exception):
    pass
class ModelNotFound(Exception):
    pass



class RUN(object):
    def __init__(self):
        CreateDirectory(ROOT)
    
    def CreateModel(self,Framework,ModelName,imgpath):
        save_path = Path(f"{ROOT}/{directorys[0]}/{subdirectory[3]}")

        if Framework == "TF1":
            if ModelName not in Tf1_PATH_TO_URL_SUFFIX.keys():
                raise ModelNotFound(f"{ModelName} is not model Zoo" )

            labempa_name = "mscoco_label_map.pbtxt"
            labelmap_path = Path(f"{ROOT}/{directorys[1]}/{subdirectory[0]}/{childdirectory[1]}/{labempa_name}")

            model_path = Path(f"{ROOT}/{directorys[1]}/{subdirectory[0]}/{childdirectory[0]}")
            model_filename = Path(f"{model_path}/{Tf1_PATH_TO_URL_SUFFIX[ModelName]}")
            dir_name = str(model_filename).split(".")[0]
            model_pb_path = glob.glob(f"{str(dir_name)}/*.pb")


            if os.path.isfile(str(model_pb_path[0])):
                Model = TF1(str(model_pb_path[0]),str(labelmap_path),imgpath,save_dir=str(save_path))
                value = Model()
                return value

            else:
                print( Fore.YELLOW,f"{ModelName} is download")
                url = ModelUrl.tf1(ModelName)
                model_tar_file = modeldownload.download(url.get,str(model_filename))
                Untar.extract(model_tar_file,str(model_path))
                model_pb_path = glob.glob(f"{str(dir_name)}/*.pb")
                Model = TF1(str(model_pb_path[0]),str(labelmap_path),save_dir=str(save_path))
                value = Model()
                return value


        elif Framework == "Detectron2":
            if f"COCO-Detection/{ModelName}" not in detecron2_PATH_TO_URL_SUFFIX.keys():
                raise ModelNotFound(f"{ModelName} is not found detectron2 model Zoo")
            
            output_name = "output.json"
            output_json = Path(f"{ROOT}/{directorys[1]}/{subdirectory[1]}/{childdirectory[2]}/{output_name}")
            config_name = Path(f"{ROOT}/{directorys[1]}/{subdirectory[1]}/{childdirectory[3]}/{ModelName}.yml")
            model_path = Path(f"{ROOT}/{directorys[1]}/{subdirectory[1]}/{childdirectory[4]}")
            model_filename = Path(f"{model_path}/{ModelName}.pkl")

            if os.path.isfile(str(model_filename)):
                Model = Detectron2(str(model_filename),str(config_name),str(imgpath),str(save_path),str(output_json))
                value = Model()
                return value

            else:
                print(Fore.MAGENTA,f"{ModelName} is download")
                url = ModelUrl.detectron2(ModelName)
                model_name_path= modeldownload.download(url.get,str(model_filename))
                Model = Detectron2(str(model_filename),str(config_name),str(imgpath),str(save_path),str(output_json))
                value = Model()
                return value

        elif Framework == "YOLOV5" :
            if ModelName not in yolov5_model_name:
                raise ModelNotFound(f"{ModelName} is not found Yolov5 model not found")

            model_path = Path(f"{ROOT}/{directorys[1]}/{subdirectory[2]}/{childdirectory[5]}")
            model_filename = Path(f"{model_path}/{ModelName}.pt")

            if os.path.isfile(str(model_filename)):
                Model = YOLO(str(model_filename),imgpath,str(save_path))
                value = Model()
                return value

            else:
                print(Fore.BLUE,f"{ModelName} is downlaod")
                url = ModelUrl.yolov5(ModelName)
                dd = modeldownload.download(url.get,str(model_filename))
                Model = YOLO(str(model_filename),imgpath,str(save_path))
                value = Model()
                return value
                


        else:
            raise Framework(f"Given {Framework} is not present")







if __name__ == '__main__':
    s = RUN()
    # s.CreateModel("TF1","ssd_mobilenet_v1_fpn_coco",'ssss.jpg')
    # s.CreateModel("Detectron2","faster_rcnn_R_50_FPN_1x","input.jpg")
    # s.CreateModel("YOLOV5","yolov5l","input.jpg")
    CreateDirectory.removeshutdown(ROOT)

