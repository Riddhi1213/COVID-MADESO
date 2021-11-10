
from flask import Flask, render_template, Response, flash, request, redirect, url_for 
from camera import VideoCamera
import urllib.request
from distancing import DistVideo
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'dist_files/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mask')
def mask():
	return render_template('mask.html')

@app.route('/upload')
# def distancing(filename):
# 	return render_template('distancing.html', filename=filename)
def upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        return render_template('distancing.html', filename=filename)


@app.route('/distancing')
def distancing(filename):
  return render_template('distancing.html', filename=filename)

@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dist_video_feed/<filename>')
def dist_video_feed(filename):
    return Response(gen(DistVideo(filename)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run( debug=False)
