from flask import Flask,request,send_from_directory
from flask_cors import CORS
import torch
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# Define the image preprocessing steps
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

IMAGE_DIR = "c:/temp/images/"  # directory where the images are stored
FILE_LIST = "c:/temp/filelist" # list of image files
FEATURE_LIST = "c:/temp/features.pt" # list of image features
topK = 3 # number of similar images to return

# Load the pre-computed image features and file list    
features_list = torch.load(FEATURE_LIST )
print(len(features_list))
with open(FILE_LIST, 'rb') as f:
    filelist = pickle.load(f)
print(len(filelist))
@app.route('/getSimilarImageFromBase64', methods=['POST'])
def getSimilarImagesFromBase64():
    imageString = request.form['base64Image']
    base64String = imageString.split(",")[1] # remove the header 'data:image/jpeg;base64,'
    
    # Convert the base64 string to an image
    f = BytesIO()
    f.write(base64.b64decode(base64String))
    f.seek(0)
    image = Image.open(f)
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    # Get the image features
    with torch.no_grad():
        queryFeature = model(image)
    
    score_list = [] # list of cosine similarity scores
    result_files = [] # list of image file names
    # Compare the query image with all the images in the dataset
    for i, feature in enumerate(features_list):
        # Compute the cosine similarity score
        score_t = cos_sim(queryFeature, feature)
        score = score_t.item()
        # Add the score to the list of scores
        if len(score_list) < topK :
            score_list.append(score)
            result_files.append(filelist[i])
        else:
            # Replace the smallest score with the current score 
            #  if current score is larger
            if min(score_list) < score:                    
                min_idx = score_list.index(min(score_list))
                score_list[min_idx] = score
                result_files[min_idx] = filelist[i]
                
    # Create a dictionary of the results
    resutltObj = {}
    for i, file in enumerate(result_files):
        resutltObj[file] = score_list[i]
        
    return resutltObj

# Download the image file
@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    IMAGE_DIR = "c:/temp/images/"
    
    return send_from_directory(IMAGE_DIR, filename, as_attachment=True,mimetype = "image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)