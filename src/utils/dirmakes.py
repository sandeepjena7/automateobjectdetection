
from pathlib import Path
from re import sub
import sys
import os
import glob
import time
from src.utils.allutils import IMG_FORMATS,VID_FORMATS

directorys = ['Workdir',"hub",'logs']
subdirectory = ['tf1','detectron2','yolo','exp']
childdirectory = ['model','labelmap','json','configyaml','modelpth_pkl','model_pt']



class CreateDirectory(object):
    def __init__(self,path):
        self.p = str(path)
        for dir_ in directorys:
            drc = Path(f"{self.p}/{dir_}") 
            drc.mkdir(parents=True, exist_ok=True)
        
        for dir_ in subdirectory[:-1]:
            subdir = Path(f"{self.p}/{directorys[1]}/{dir_}")
            subdir.mkdir(parents=True, exist_ok=True)
        
        subdir = Path(f"{self.p}/{directorys[0]}/{subdirectory[-1]}")
        subdir.mkdir(parents=True, exist_ok=True)

        for dir_ in childdirectory:

            if dir_ in childdirectory[:2]:
                drc = Path(f"{self.p}/{directorys[1]}/{subdirectory[0]}/{dir_}")
                drc.mkdir(parents=True, exist_ok=True)
            elif dir_ in childdirectory[2:5]:
                drc = Path(f"{self.p}/{directorys[1]}/{subdirectory[1]}/{dir_}")
                drc.mkdir(parents=True, exist_ok=True)
            elif dir_ in childdirectory[-1]:
                drc = Path(f"{self.p}/{directorys[1]}/{subdirectory[2]}/{dir_}")
                drc.mkdir(parents=True, exist_ok=True)

                


    @staticmethod
    def removefileuser(path):
        p = str(path)
        workdir = Path(f"{p}/{directorys[0]}")
        assert os.path.isdir(str(workdir)) ,'workdir is not present'

        for file in workdir.iterdir():
            if os.path.isfile(str(file)):
                Path(file).unlink()

            elif os.path.isdir(str(file)):
                for fl in file.iterdir():
                    Path(fl).unlink()

    @staticmethod
    def removeshutdown(path):
        p = str(path)
        # woking dir

        workdir = Path(f"{p}/{directorys[0]}")
        assert os.path.isdir(str(workdir)) ,'workdir is not present'
        for file in workdir.iterdir():
            print(file)
            if os.path.isfile(str(file)):

                Path(file).unlink()

            elif os.path.isdir(str(file)):
                for fl in file.iterdir():
                    Path(fl).unlink()
                
                Path(file).rmdir()
            # time.sleep(1)
        Path(workdir).rmdir()
        
        # hub

        hub = Path(f"{p}/{directorys[1]}")
        assert os.path.isdir(str(hub)) ,'some problem occurred hub'

        # tf1
        tf1_subdir = Path(f"{hub}/{subdirectory[0]}")
        assert os.path.isdir(str(hub)) ,'some problem occurred tf1'
        tf1_child = Path(f"{tf1_subdir}/{childdirectory[0]}")
        assert os.path.isdir(str(tf1_child)) ,'some problem occurred tf1 model directroy'

        for tf_1files in Path(tf1_child).rglob("*.*"):
            Path(tf_1files).unlink()

        for tf_1dirfiles in tf1_child.iterdir():
            if os.path.isdir(tf_1dirfiles):

                for saved_model in tf_1dirfiles.iterdir():
                    if os.path.isfile(saved_model):
                        Path(saved_model).unlink()

                        pass
                    elif os.path.isdir(saved_model):
                        for varibles in saved_model.iterdir():
                            Path(varibles).rmdir()

                        Path(saved_model).rmdir()
                Path(tf_1dirfiles).rmdir()
        
        # yolo
        yolo_subdir = Path(f"{hub}/{subdirectory[2]}")
        assert os.path.isdir(str(hub)) ,'some problem occurred yolo'
        yolo_modeldir = Path(f"{yolo_subdir}/{childdirectory[-1]}")
        for yolomodle in Path(yolo_modeldir).rglob("*.*"):
            if os.path.isfile(str(yolomodle)):
                Path(yolomodle).unlink()

        # detectron2
        detectron2_subdir = Path(f"{hub}/{subdirectory[1]}")
        assert os.path.isdir(str(hub)) ,'some problem occurred yolo'
        detectron2_modeldir = Path(f"{detectron2_subdir}/{childdirectory[-2]}")
        for detectron2model in Path(detectron2_modeldir).rglob("*.*"):
            if os.path.isfile(detectron2model):
                Path(detectron2model).unlink()



            





    

if __name__ == '__main__':
    s = Path(__file__).resolve().parent

    m = CreateDirectory(s)
    # # j = CreateDirectory.removefileuser(s)
    l = CreateDirectory.removeshutdown(s)

