from pathlib import Path
import sys
from src.utils.allutils import IMG_FORMATS,VID_FORMATS
import os
import glob
import time


class WorkDir:
    def __init__(self,path,mode=None):

        self.path = str(path) 
        workdir_name = 'workdir'
        imagedir_name = "images"
        videodir_name = "videos"
        inputimage_name = f"inputimage.{IMG_FORMATS[2]}"
        inputvideo_name = f"inputvideo.{VID_FORMATS[6]}"
        infercencedir_name = "infercence"
     
        self.workdir = Path(f"{self.path}/{workdir_name}")
        self.workdir.mkdir(parents=True, exist_ok=True)


        if mode == "image":
            imgaesdir = Path(f"{self.workdir}/{imagedir_name}")
            inputimage_file = Path(f"{imgaesdir}/{inputimage_name}")
            infercencedir = Path(f"{self.workdir}/{infercencedir_name}")

            infercencedir.mkdir(parents=True,exist_ok=True)
            imgaesdir.mkdir(parents=True, exist_ok=True)
            open(inputimage_file,mode='w').close()


        elif mode == "video":
            videodir = Path(f"{self.workdir}/{videodir_name}")
            inputvideo_file = Path(f"{videodir}/{inputvideo_name}")
            infercencedir = Path(f"{self.workdir}/{infercencedir_name}")

            infercencedir.mkdir(parents=True,exist_ok=True)
            videodir.mkdir(parents=True, exist_ok=True)
            open(inputvideo_file,mode='w').close()

        else:
            imgaesdir = Path(f"{self.workdir}/{imagedir_name}")
            inputimage_file = Path(f"{imgaesdir}/{inputimage_name}")


            imgaesdir.mkdir(parents=True, exist_ok=True)
            open(inputimage_file,mode='w').close()


            videodir = Path(f"{self.workdir}/{videodir_name}")
            inputvideo_file = Path(f"{videodir}/{inputvideo_name}")


            videodir.mkdir(parents=True, exist_ok=True)
            open(inputvideo_file,mode='w').close()
            infercencedir = Path(f"{self.workdir}/{infercencedir_name}")

            infercencedir.mkdir(parents=True,exist_ok=True)

    def __del__(self):
        if os.path.isdir(self.workdir):
            for file in (sorted(glob.glob(os.path.join(self.workdir, '*/*')))): Path(file).unlink()
            for directory in (glob.glob(f"{self.workdir}/*")): Path(directory).rmdir()
            Path(self.workdir).rmdir()





    

if __name__ == '__main__':
    s = Path(__file__).resolve().parent
    # print(str(s))
    # print(VID_FORMATS[6])
    m = WorkDir(s)

    # with open('video.mp4',"w") as f: pass
    # # open("video.mp4", mode='w').close()