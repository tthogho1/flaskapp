from flask import Flask, render_template,request,send_from_directory
from util.ImageSearch import ImageSearch

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getSimilarImage', methods=['POST'])
def getSimilarImages():
    imsearch = ImageSearch()
    file = request.files['file']
    result_files, score_list = imsearch.searchImages(file)
    
    resutltObj = {}
    for i, file in enumerate(result_files):
        resutltObj[file] = score_list[i]
        
    return resutltObj

@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    IMAGE_DIR = "c:/temp/images/"
    #print(filename)
    
    return send_from_directory(IMAGE_DIR, filename, as_attachment=True,mimetype = "image/jpeg")


if __name__ == "__main__":
    app.run(debug=True)