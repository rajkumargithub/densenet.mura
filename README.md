# Abnormality Detection in Musculoskeletal Radiographs using DenseNet169 (Keras)

You can use the [editor on GitHub](https://github.com/rajkumargithub/rajkumar.github.io/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

## Objective
The objective is to build a machine learning model that can detect an abnormality in the X-Ray radiographs. These models can help towards providing healthcare access to the parts of the world where access to skilled radiologists is limited. According to a study on the Global Burden of Disease and the worldwide impact of all diseases found that, “musculoskeletal conditions affect more than 1.7 billion people worldwide. They are the 2nd greatest cause of disabilities, and have the 4th greatest impact on the overall health of the world population when considering both death and disabilities”. (www.usbji.org, n.d.).

Stanford University Machine Learning Group has published a paper related to this problem and provided one of the world’s largest public radiographic images dataset called MURA. MURA is short for Musculoskeletal Radiographs. Stanford university ML group has used DenseNet169 algorithm to train a deep neural network which can detect abnormalities in the radiographs with the accuracy closer to the top radiologists. This project attempts to implement deep neural network using DenseNet169 inspired from the Stanford Paper (Rajpurkar, et al., 2018).

## Dataset
As per the paper “MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs”, the data is collected de-identified, HIPAA-compliant images from the Pictures Archive and Communication System (PACS) of Stanford Hospital. It consists of 14,863 studies from 12,173 patients, with a total of 40,561 multi-view radiographic images. Each belongs to one of seven Stanford upper extremity radiographic study types: elbow, finger, forearm, humerus, shoulder and wrist. Each study was manually labeled as normal or abnormal by board-certified radiologists from the Stanford Hospital at the time of clinical radiographic interpretation in the diagnostic radiology environment between 2001 and 2012. The labeling was performed during interpretation on DICOM images presented on at least 3 megapixel PACS medical grade display and max luminance 400 cd/m2 and min luminance 1 cd/m2 with pixel size of 0.2 and native resolution of 1500 x 2000 pixels. The clinical images vary in resolution and in aspect ratios. The dataset has 9,045 normal and 5,818 abnormal musculoskeletal radiographic studies. (Rajpurkar, et al., 2018)

This dataset is made available to the community to encourage research on the healthcare domain and build a much better performing model. This dataset can be downloaded from Stanford website.

### Markdown

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
