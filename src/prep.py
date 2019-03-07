import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
import os
import cv2


def create_studies_metadata_csv(category):
    """
    This function creates a csv file containing the path of studies, count of images & label.
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    study_types = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']
    i = 0
    study_data[category] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
    for study_type in study_types: # Iterate throught every study types
        DATA_DIR = '../data/MURA-v1.1/%s/%s/' % (category, study_type)
        patients = list(os.walk(DATA_DIR))[0][1]  # list of patient folder names
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(DATA_DIR + patient):  # for each study in that patient folder
                if(study != '.DS_Store'):
                    label = study_label[study.split('_')[1]]  # get label 0 or 1
                    path = DATA_DIR + patient + '/' + study + '/'  # path to this study
                    study_data[category].loc[i] = [path, len(os.listdir(path)), label]  # add new row
                    i += 1
    study_data[category].to_csv("../data/"+category+"_study_data.csv",index = None, header=False)

def create_images_metadata_csv(category):
    """
    This function creates a csv file containing the path of images, label.
    """
    image_data = {}
    study_label = {'positive': 1, 'negative': 0}
    study_types = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']
    i = 0
    image_data[category] = pd.DataFrame(columns=['Path', 'Label'])
    for study_type in study_types: # Iterate throught every study types
        DATA_DIR = '../data/MURA-v1.1/%s/%s/' % (category, study_type)
        patients = list(os.walk(DATA_DIR))[0][1]  # list of patient folder names
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(DATA_DIR + patient):  # for each study in that patient folder
                if(study != '.DS_Store'):
                    label = study_label[study.split('_')[1]]  # get label 0 or 1
                    path = DATA_DIR + patient + '/' + study + '/'  # path to this study
                    for j in range(len(os.listdir(path))):
                        image_path = path + 'image%s.png' % (j + 1)
                        image_data[category].loc[i] = [image_path, label]  # add new row
                        i += 1
    image_data[category].to_csv("../data/"+category+"_image_data.csv",index = None, header=False)

def create_studytype_classes_csv(category):
    """
    This function creates a csv file containing the path of studies, count of images & label.
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    study_types = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']
    i = 0


    study_data[category] = pd.DataFrame(columns=['StudyType', 'Patient', 'Study','Label'])
    for study_type in study_types: # Iterate throught every study types
        DATA_DIR = '../data/MURA-v1.1/%s/%s/' % (category, study_type)
        patients = list(os.walk(DATA_DIR))[0][1]  # list of patient folder names
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(DATA_DIR + patient):  # for each study in that patient folder
                if(study != '.DS_Store'):
                    label = study_label[study.split('_')[1]]  # get label 0 or 1
                    study_data[category].loc[i] = [study_type, patient, study, label]  # add new row
                    i += 1
    study_data[category].to_csv("../data/"+category+"_studytype_classes.csv",index = None, header=False)


#create_studies_metadata_csv('train')
#create_studies_metadata_csv('valid')

#create_images_metadata_csv('train')
#create_images_metadata_csv('valid')


create_studytype_classes_csv('train')
create_studytype_classes_csv('valid')
