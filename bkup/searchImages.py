import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device= get_device()
# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True).to(device)
model.eval()
# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#
# Load the input image and apply the preprocessing
#
IMAGE_DIR = "c:/temp/images/"
FILE_LIST = "c:/temp/filelist"
FEATURE_LIST = "c:/temp/features.pt"
filelist = os.listdir(IMAGE_DIR)

features_list = torch.load(FEATURE_LIST )
# 非pickle化
with open(FILE_LIST, 'rb') as f:
    filelist = pickle.load(f)

def getFeature(imgFile):
    image = Image.open(imgFile)
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)

    return features

cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
def searchImages(queryImage, features_list, topK=5):
    queryFeature = getFeature(queryImage)
    
    score_list = []
    result_files = []
    for i, feature in enumerate(features_list):
        score = cos_sim(queryFeature, feature)
        print(score)
        print(filelist[i])
        if len(score_list) < topK:
            score_list.append(score)
            result_files.append(filelist[i])
        else:
            if min(score_list) < score:                    
                min_idx = score_list.index(min(score_list))
                score_list[min_idx] = score
                result_files[min_idx] = filelist[i]
                
    # Return the top-k images and their scores
    return result_files, score_list

files, score = searchImages("c:/temp/images/1000550952.jpg",features_list)

print(files)
print(score)