# **Abnormality Detection in Musculoskeletal Radiographs using [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)**

## Objective
The objective is to build a machine learning model that can detect an abnormality in the X-Ray radiographs. These models can help towards providing healthcare access to the parts of the world where access to skilled radiologists is limited. According to a study on the Global Burden of Disease and the worldwide impact of all diseases found that, “musculoskeletal conditions affect more than 1.7 billion people worldwide. They are the 2nd greatest cause of disabilities, and have the 4th greatest impact on the overall health of the world population when considering both death and disabilities”. (www.usbji.org, n.d.).

Stanford University Machine Learning Group has published a paper related to this problem and provided one of the world’s largest public radiographic images dataset called MURA. MURA is short for Musculoskeletal Radiographs. Stanford university ML group has used DenseNet169 algorithm to train a deep neural network which can detect abnormalities in the radiographs with the accuracy closer to the top radiologists. This project attempts to implement deep neural network using DenseNet169 inspired from the Stanford Paper [Rajpurkar, et al., 2018](https://arxiv.org/abs/1712.06957).

##Dataset
As per the paper “MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs”, the data is collected de-identified, HIPAA-compliant images from the Pictures Archive and Communication System (PACS) of Stanford Hospital. It consists of 14,863 studies from 12,173 patients, with a total of 40,561 multi-view radiographic images. Each belongs to one of seven Stanford upper extremity radiographic study types: elbow, finger, forearm, humerus, shoulder and wrist. Each study was manually labeled as normal or abnormal by board-certified radiologists from the Stanford Hospital at the time of clinical radiographic interpretation in the diagnostic radiology environment between 2001 and 2012. The labeling was performed during interpretation on DICOM images presented on at least 3 megapixel PACS medical grade display and max luminance 400 cd/m2 and min luminance 1 cd/m2 with pixel size of 0.2 and native resolution of 1500 x 2000 pixels. The clinical images vary in resolution and in aspect ratios. The dataset has 9,045 normal and 5,818 abnormal musculoskeletal radiographic studies. [Rajpurkar, et al., 2018](https://arxiv.org/abs/1712.06957).

### Data Collection
Stanford review board approved study collected de-idenified, HIPAA-compliant images from the Picture Archive and Communication Systems (PACS) for Stanford Hospital. The assembled a dataset of musculoskeletal radiographs consists of 14,863 studies from 12,173 patients, with the total of 40,561 multi-view radiographic images. Each belongs to one of seven standard upper extremity radiographic study types: elbow, finger, forearm, hand, humerus, shoulder, and wrist. Each study was manually labeled as normal or abnormal by board-certified radiologist from the Stanford Hospital at the time of clinical radiographic interpretation in the diagnostic radiology environment between 2001 and 2012. The labeling was performed during interpretation on DICOM images presented on at least 3 megapixel PACS medical grade display with max luminance 400 cd/m<sup>2</sup> and min luminance 1 cd/m<sup>2</sup> with pixel size of 0.2 and naive resolution of 1500 x 2000 pixels. The clinial images vary in resolution and in aspect ratios. The dataset has been split into training (11,184 patients, 13,457 studies, 36,808 images), validation (783 patients, 1,199 studies, 3,197 images), and test (206 patients, 207 studies, 556 images) sets. There is no overlap in patients between any of the sets. To evaluate the models and get a robust estimate of radiologist performance, additional labels from the board-certified Stanford radiologists on the test set, consisting of 207 musculoskeletal studies. The radiologists individually retrospectively reviewed and labeled each study in the test set as a DICOM file as normal or abnormal in the clinical reading room environment using the PACS. The radiologists have 8.83 years of experience on average ranging from 2 to 25 years. The radiologists did not have access to any clinical information. Labels were entered into a standardized data entry program.[Rajpurkar, et al., 2018](https://arxiv.org/abs/1712.06957).

### Abnormality Analysis
To investigate the types of abnormalities present in the dataset, we reviewed the radiologist reports to manually label 100 abnormal studies with the abnormality finding: 53 studies were labeled with fractures, 48 with hardware, 35 with degenerative join diseases, and 29 with other miscellaneous abnormalities, including lesions and subluxations.[Rajpurkar, et al., 2018](https://arxiv.org/abs/1712.06957).

## Exploratory Data analysis
I have created a 2 types of metadata. One containing information for each image with its corresponding label & study type. Another one information containing for each study type, label & image count. So that i can plot the data to understand various characteristics. Also, i have plotted some images from both the classes (Normal & Abnormal) to visualize it.

```
fig, ax = plt.subplots(figsize=(10, 5))
for i, phase in enumerate(data_cat):
    counts = patients_count[phase].values()
    m = max(counts)
    for i, v in enumerate(counts):
        if v==m: ax.text(i-0.1, v+3, str(v))
        else: ax.text(i-0.1, v + 20, str(v))
    x_pos = np.arange(len(study_types))
    plt.bar(x_pos, counts, alpha=0.5)
    plt.xticks(x_pos, study_types)

plt.xlabel('Study types')
plt.ylabel('Number of patients')
plt.legend(['train', 'valid'])
plt.show()
fig.savefig('../images/pcpst.jpg', bbox_inches='tight', pad_inches=0)
```

![image](images/pcpst.jpg)</center>
<center>Figure 1 [Rishabh Agrahari, 2018](https://medium.com/@pyaf/implementing-densenet-on-mura-using-pytorch-f39e92566815)

Figure 1 shows data from each study types. XR_WRIST has largest number of data compared to other study types. But one important thing is that, there is no significant variation in the validation data volumes among on the study types. It may cause some issue when predicting the classes for each study types. Because some study types don't have enough training data compared to others. For example _"XR-HUMERUS"_

![image](https://github.com/rajkumargithub/densenet.mura/blob/master/images/pcpsc.jpg?raw=true)</center>
<center>Figure 2 [Rishabh Agrahari, 2018](https://medium.com/@pyaf/implementing-densenet-on-mura-using-pytorch-f39e92566815)

Figure 2 shows the distribution of studies counts in each study (patient level). Sometimes a diagnosis requires multiple studies of different radiographs. It shows how many patients have to under-go multiple studies. It seems like more than 90% of the time, a patient require just one study. However, sometimes they under-go multiple studies.

![image](https://github.com/rajkumargithub/densenet.mura/blob/master/images/pcplc.jpg?raw=true)</center>
<center>Figure 3 [Rishabh Agrahari, 2018](https://medium.com/@pyaf/implementing-densenet-on-mura-using-pytorch-f39e92566815)

## Pre-preprocessing
As per the paper, i have normalized the each image to have same mean & std of the images in the ImageNet training set. the scaled the images. In the paper, they have used variable-sized images to 320 x 320. But i have chosen to scale 224 x 224.  Then i have augmented the data during the training by applying random lateral inversions and rotations of up to 30 degrees using `ImageDataGenerator`

```
img = cv2.imread(data['Path']) #read the image
img = cv2.resize(img,(img_width,img_height))  #scale to 224 x 224  
img = img[...,::-1].astype(np.float32) #convert the array into float32 dtype
```
```
mean = np.mean(images[:, :, :]) #normalization
std = np.std(images[:, :, :])
images[:, :, :] = (images[:, :, :] - mean) / std
```

```
train_datagen = ImageDataGenerator( #rotations & lateral inversions
    rotation_range=30,
    horizontal_flip=True
    )

test_datagen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True
    )
```

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/rajkumargithub/rajkumar.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
