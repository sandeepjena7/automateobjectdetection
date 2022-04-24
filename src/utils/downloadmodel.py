import functools
import pathlib
import shutil
import requests
from tqdm.auto import tqdm
from colorama import Fore
import tarfile
from src.utils.allutils import detecron2_PATH_TO_URL_SUFFIX,Tf1_PATH_TO_URL_SUFFIX



class ModelUrl:
    def __init__(self,url=None):
        self.get = url
             

    @classmethod
    def yolov5(cls,modelname):
        get =  f"https://github.com/ultralytics/yolov5/releases/download/v6.1/{modelname}.pt"
        return cls(get)

    @classmethod
    def detectron2(cls,modelname):
        S3_PREFIX = "https://dl.fbaipublicfiles.com/detectron2"
        model = f"COCO-Detection/{modelname}"
        detecron2_URL_SUFFIX = detecron2_PATH_TO_URL_SUFFIX[model]
        get = f"{S3_PREFIX}/{model}/{detecron2_URL_SUFFIX}"
        return cls(get)

    @classmethod
    def tf1(cls,modelname):
        S3_PREFIX = "http://download.tensorflow.org/models/object_detection"

        tf1_URL_SUFFIX = Tf1_PATH_TO_URL_SUFFIX[modelname]
        get = f"{S3_PREFIX}/{tf1_URL_SUFFIX}"
        return cls(get)




class modeldownload(object):
    @staticmethod
    def download(url, filename):
        #  https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
        
        r = requests.get(url, stream=True, allow_redirects=True)
        if r.status_code != 200:
            r.raise_for_status()  # Will only raise for 4xx codes, so...
            raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
        file_size = int(r.headers.get('Content-Length', 0))

        path = pathlib.Path(filename).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        desc = "(Unknown total file size)" if file_size == 0 else ""
        r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
            with path.open("wb") as f:
                shutil.copyfileobj(r_raw, f)

        print(Fore.WHITE,end='')
        return path



class Untar(object):
    @staticmethod
    def extract(tarfilepath,desintationpath):
        print(Fore.CYAN,end='')

        with tarfile.open(name=tarfilepath) as tar:

            # Go over each member
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):

                # Extract member
                tar.extract(member,desintationpath)
        print(Fore.WHITE)


