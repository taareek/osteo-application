import io 
import base64
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
import pickle
import datetime
import numpy as np
from collections import Counter
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torchvision.models import densenet121, DenseNet121_Weights

# CNN model as feature extractor 
# Taking pre-trained DenseNet-121 model and freezing layers to use pre-trained weights
# DEFAULT weights give us best available weights 
weights = DenseNet121_Weights.DEFAULT
model = densenet121(weights= weights)
for param in model.parameters():
        param.requires_grad = False

# we will extract 1024 features from densnet pre trained model
model.classifier = nn.Linear(1024, 1024)

# load pre-trained model 
model.load_state_dict(torch.load('./mlp_classifier/web_app_1024.pt', map_location=torch.device('cpu')))

# if we use GPU
# if torch.cuda.is_available:
#     model.cuda()
# else:
#     pass

# set model as evaluation mode
model.eval()

# loading saved ML Classifier 
# classifier_path = "./mlp_classifier/without_fs_mlp.pkl"
# pickle_in = open(classifier_path, "rb")
# classifier = pickle.load(pickle_in)

# loading all pre-trained MLP classifier
# multiclss classifier 
mul_cls_path = "./mlp_classifier/multiclass_rfe_mlp.pkl"
pickle_in = open(mul_cls_path, "rb")
multiclass_classifier = pickle.load(pickle_in)

# non-tumor vs necrosis+viable classifier 
bn1_cls_path = "./mlp_classifier/nt_vs_nec_plus_via_rfe_mlp.pkl"
pickle_in = open(bn1_cls_path, "rb")
nt_vs_nec_plus_via_classifier = pickle.load(pickle_in)

# non-tumor vs necrosis
bn2_cls_path = "./mlp_classifier/nt_vs_nec_rfe_mlp.pkl"
pickle_in = open(bn2_cls_path, "rb")
nt_vs_nec_classifier = pickle.load(pickle_in)

# non-tumor vs viable 
bn3_cls_path = "./mlp_classifier/nt_vs_via_rfe_mlp.pkl"
pickle_in = open(bn3_cls_path, "rb")
nt_vs_via_classifier = pickle.load(pickle_in)

# necrosis vs viable 
bn4_cls_path = "./mlp_classifier/nec_vs_via_rfe_mlp.pkl"
pickle_in = open(bn4_cls_path, "rb")
nec_vs_via_classifier = pickle.load(pickle_in)

# loading all saved feature selector 
# getting multiclass feature selector pickle file
feats_path = "./mlp_classifier/fs_multiclass_rfe.pkl"
feats_in = open(feats_path, "rb")
multiclass_ft_selector = pickle.load(feats_in)

# non-tumor vs necrosis+viable 300
bn1_ft_path = "./mlp_classifier/fs_nt_vs_nec_plus_via_rfe.pkl"
feats_in = open(bn1_ft_path, "rb")
nt_vs_nec_plus_via_ft_selector = pickle.load(feats_in)

# non-tumor vs necrosis 500
bn2_ft_path = "./mlp_classifier/fs_nt_vs_nec_rfe.pkl"
feats_in = open(bn2_ft_path, "rb")
nt_vs_nec_selector = pickle.load(feats_in)

# non-tumor vs viable 600
bn3_ft_path = "./mlp_classifier/fs_nt_vs_via_rfe.pkl"
feats_in = open(bn3_ft_path, "rb")
nt_vs_via_selector = pickle.load(feats_in)

# necrosis vs viable 300
bn4_ft_path = "./mlp_classifier/fs_nec_vs_via_rfe.pkl"
feats_in = open(bn4_ft_path, "rb")
nec_vs_via_selector = pickle.load(feats_in)

#defining our class index
class_index = ["Non-tumor", "Necrosis", "Viable"]

# defining feature scaler
sc = StandardScaler()

# applying transformation on input image (resize, torch, normalization)
def image_transformation(input_img):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    data_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
            ])

    image = Image.open(io.BytesIO(input_img))
    image = data_transforms(image).unsqueeze(0)
    return image

# perform feature extraction
def get_features(input_image):
    img_to_tensor = image_transformation(input_image)

    # feed img tensor into CNN to get the features 
    img_features = model.forward(img_to_tensor)

    # converting to numpy 
    if torch.cuda.is_available():
        img_features = img_features.cpu().detach().numpy()   # if use GPU
    else:
        img_features = img_features.detach().numpy()
    
    return img_features

# perform feature selection and classification 
# multi class classifiaction
def multiclass_prediction(img_features):
  # perform feature selection for multiclass 
  multiclass_fs = multiclass_ft_selector.transform(img_features)
  # multiclass classification 
  multiclass_pred = multiclass_classifier.predict(multiclass_fs)
  # multiclass class probability 
  multiclass_pred_proba = multiclass_classifier.predict_proba(multiclass_fs)
  multiclass_pred_proba = np.max(multiclass_pred_proba) 
  multiclass_pred_proba = float("{:.2f}".format(multiclass_pred_proba))

  return multiclass_pred[0], multiclass_pred_proba*100

# binary classification_1 non-tumor vs nec+viable feature selection 
def nt_versus_nec_plus_via(img_features):
  # feature selection
  nt_vs_nec_plus_via_fs = nt_vs_nec_plus_via_ft_selector.transform(img_features)
  # perform non-tumor vs nec+viable classification
  bn_1_pred = nt_vs_nec_plus_via_classifier.predict(nt_vs_nec_plus_via_fs)
  # class probability 
  bn_1_pred_proba = nt_vs_nec_plus_via_classifier.predict_proba(nt_vs_nec_plus_via_fs)
  bn_1_pred_proba = np.max(bn_1_pred_proba)
  bn_1_pred_proba = float("{:.2f}".format(bn_1_pred_proba))

  return bn_1_pred[0], bn_1_pred_proba*100

# binary classification_2  vs non-tumor vs necrosis feature selection 
def nt_versus_nec(img_features):
  # feature selection
  nt_vs_nec_fs = nt_vs_nec_selector.transform(img_features)
  # viable vs non-tumor classification
  bn_2_pred = nt_vs_nec_classifier.predict(nt_vs_nec_fs)
  # class probability 
  bn_2_pred_proba = nt_vs_nec_classifier.predict_proba(nt_vs_nec_fs)
  bn_2_pred_proba = np.max(bn_2_pred_proba)
  bn_2_pred_proba = float("{:.2f}".format(bn_2_pred_proba))
  
  return bn_2_pred[0], bn_2_pred_proba*100

# binary classification_4 non-tumor vs viable feature selectio
def nt_versus_via(img_features):
  # feature selection
  nt_vs_via_fs = nt_vs_via_selector.transform(img_features)
  # non-tumor versus viable classification 
  bn_3_pred = nt_vs_via_classifier.predict(nt_vs_via_fs)
  # class probability 
  bn_3_pred_proba = nt_vs_via_classifier.predict_proba(nt_vs_via_fs)
  bn_3_pred_proba = np.max(bn_3_pred_proba)
  bn_3_pred_proba = float("{:.2f}".format(bn_3_pred_proba))

  return bn_3_pred[0], bn_3_pred_proba*100

#binary classification_3 necrosis vs viable feature selection
def nec_versus_via(img_features):
  # feture selection 
  nec_vs_via_fs = nec_vs_via_selector.transform(img_features)
  # nec vs viable classification 
  bn_4_pred = nec_vs_via_classifier.predict(nec_vs_via_fs)
  # class probability 
  bn_4_pred_proba = nec_vs_via_classifier.predict_proba(nec_vs_via_fs)
  bn_4_pred_proba = np.max(bn_4_pred_proba)
  bn_4_pred_proba = float("{:.2f}".format(bn_4_pred_proba))

  return bn_4_pred[0], bn_4_pred_proba*100

def prediction(img_features):
  # to store all predicted class and class probabilities 
  class_list = []
  proba_list = []
  # perform multiclass classification 
  multiclass_pred, multiclass_proba = multiclass_prediction(img_features)
  # append to list
  class_list.append(multiclass_pred)
  proba_list.append(multiclass_proba)

  # perform binaray classification-1: non-tumor vs necrosis+viable  
  bn_1_pred, bn_1_pred_proba = nt_versus_nec_plus_via(img_features)
  # append to list
  class_list.append(bn_1_pred)
  proba_list.append(bn_1_pred_proba)

  # perform binaray classification-2: non-tumor vs necrosis   
  bn_2_pred, bn_2_pred_proba = nt_versus_nec(img_features)
  # append to list
  class_list.append(bn_2_pred)
  proba_list.append(bn_2_pred_proba)

  # perform binaray classification-3: non-tumor vs viable  
  bn_3_pred, bn_3_pred_proba = nt_versus_via(img_features)
  # append to list
  class_list.append(bn_3_pred)
  proba_list.append(bn_3_pred_proba)

  # perform binaray classification-4: necrosis vs viable  
  bn_4_pred, bn_4_pred_proba = nec_versus_via(img_features)
  # append to list
  class_list.append(bn_4_pred)
  proba_list.append(bn_4_pred_proba)

  return class_list, proba_list

# function to find most frequent value in a list 
def most_appeared(a_list):
  occurance_count = Counter(a_list)
  freq_value = occurance_count.most_common(1)[0][0]
  frequency = occurance_count.most_common(1)[0][1]
  return freq_value, frequency

# get indexes for most frequent value
def get_indexes(freq_value, class_list):
  index_value = [index for index in range(len(class_list)) if class_list[index] == freq_value]
  return index_value

# find max probability 
def find_max_proba(index_list, proba_list, max_proba=0):
  for i in range(len(index_list)):
    k = index_list[i]
    if(proba_list[k] >= max_proba):
      max_proba = proba_list[k]
  return max_proba

# feed image features into MLP classifier
def final_prediction(features):
    # feature scaling
    # feature selection
    class_list, proba_list = prediction(features)
    # get the most predicted class from all the classifications both binary and multiclass 
    mostly_predicted, frequency = most_appeared(class_list)
    # get the index numbers of most appeared class  
    mostly_predicted_indices = get_indexes(mostly_predicted, class_list)
    # get maximum probability
    max_probability = find_max_proba(mostly_predicted_indices, proba_list)

    # cls_prediction = classifier.predict(features)
    # cls_proba = classifier.predict_proba(features)
    # cls_proba = np.max(cls_proba) 
    # cls_proba = float("{:.2f}".format(cls_proba))
    # predicted_class = class_index[cls_prediction[0]]
    predicted_class = class_index[mostly_predicted]
    return predicted_class, max_probability


def get_result(img_file, is_api = False):
    start_time = datetime.datetime.now()
    img_bytes = img_file.file.read()
    g_features = get_features(img_bytes)
    # getting predicted class and probability
    pred_class, pred_proba = final_prediction(g_features)
    # pred_proba = pred_proba * 100
    # class_name = prediction(img_bytes)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = f'{round(time_diff.total_seconds() * 1000)} ms'
    encoded_string = base64.b64encode(img_bytes)
    bs64 = encoded_string.decode('utf-8')
    image_data = f'data:image/jpeg;base64,{bs64}'
    
    result = {
        "inference_time": execution_time,
        "prediction": {
            "class_name": pred_class,
            "class_probability": pred_proba
        }
        }

    if not is_api: 
        result["image_data"]= image_data

    return result
    

