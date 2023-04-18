import torch
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle

# Load the pre-trained ResNet50 model

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device= get_device()
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def getFeature(imgFile):
    image = Image.open(imgFile)
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)

    return features
#
# Load the input image and apply the preprocessing
#
IMAGE_DIR = "c:/temp/images/"
FILE_LIST = "c:/temp/filelist"
FEATURE_LIST = "c:/temp/features.pt"
filelist = os.listdir(IMAGE_DIR)

file_list=[]
feature_list = []
for file in filelist:
    print(file)
    features = getFeature(IMAGE_DIR + file)
    feature_list.append(features)
    file_list.append(file)

torch.save(feature_list, FEATURE_LIST)
with open(FILE_LIST,'wb') as f:
    pickle.dump(file_list, f)


