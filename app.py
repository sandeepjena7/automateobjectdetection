from flask import Flask, render_template, request, redirect, send_file, url_for,jsonify
from werkzeug.utils import secure_filename, send_from_directory
from  pathlib import Path
from src.utils.dirmakes import CreateDirectory
from run import RUN
import os

run = RUN()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

webapp_path = Path(f"{ROOT}/webapp")

static_dir = Path(f"{webapp_path}/static")
templates_dir = Path(f"{webapp_path}/templates")

app = Flask(__name__,static_folder=str(static_dir),template_folder=str(templates_dir)) # sunny sir


@app.route("/")
def hello_world():
    return render_template('new.html')


if __name__ == "__main__":
    port = 8080
    app.run( port=port)
