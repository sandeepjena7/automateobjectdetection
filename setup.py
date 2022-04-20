from setuptools import setup
import os

setup(
    name="src",
    version="0.0.1",
    # description="invisible you in video without any proof",
    # # url="https://github.com/sandeepjena7",
    # author="sandeepjean777",
    # author_email="sandeepjean777@gmail.com",
    packages=["src"],
    python_requires=">=3.6",
    install_requires=[
        "protobuf"
        ,"six"
        ,"tensorboard==1.14.0"
        ,"tensorflow==1.14.0"
        ,"opencv-python"
        ,"matplotlib"
        ,"Cython"
        ,"torch==1.8.1 "
        ,"torchvision==0.9.1"
        ,"fvcore"
        ,"cloudpickle"
        ,"omegaconf"
        ,"pandas"
        ,"seaborn"
        ,"requests"
        ,"pycocotools-windows" #ihave not install pycocotools-windows through setup.py we can install mannualy
        # pip install  pycocotools-windows
    ]
    )