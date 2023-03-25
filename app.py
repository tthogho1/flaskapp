from flask import Flask, render_template,request,send_from_directory,make_response
from util.ImageSearch import ImageSearch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getSimilarImage', methods=['POST'])
def getSimilarImages():

    file = request.files['file']
    imsearch = ImageSearch()
    result_files, score_list = imsearch.searchImagesFromFile(file)
    
    resutltObj = {}
    for i, file in enumerate(result_files):
        resutltObj[file] = score_list[i]
        
    return resutltObj


@app.route('/getSimilarImageFromBase64', methods=['POST'])
def getSimilarImagesFromBase64():
    imageString = request.form['base64Image']
    base64 = imageString.split(",")[1]
    
    imsearch = ImageSearch()
    result_files, score_list = imsearch.searchImagesFromBase64(base64)
    
    resutltObj = {}
    for i, file in enumerate(result_files):
        resutltObj[file] = score_list[i]
        
    return resutltObj


@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    IMAGE_DIR = "c:/temp/images/"
    #print(filename)
    
    return send_from_directory(IMAGE_DIR, filename, as_attachment=True,mimetype = "image/jpeg")


@app.route('/test', methods=['POST'])
def gettext():
    a = request.form['prompt'] 
    response = make_response(a, 200)
    response.mimetype = "text/plain"
    return response


if __name__ == "__main__":
    app.run(debug=True)