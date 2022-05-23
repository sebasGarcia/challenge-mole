# Computer Vision project - Mole detection with TensorFlow. Deployment with Streamlit and Heroku

## Description

The health care company "skinCare" would like a tool that is able to detect moles that need to be handled by a doctor. 
This AI tool detects the type of mole when a picture is uploaded on the web page and shows the accuracy the model has.
Since this is a Proof of Concept, the design of the website is simple but functional and runs on Heroku.


![skincare (GIF)](https://media.giphy.com/media/MCRrrB2WeQcYxn55MC/giphy.gif)

### Workflow 

The workflow of this project can be illustrated with the image below.

![ML Workflow](https://hazaq.me/assets/images/ml-workflow.jpeg)

1. Get Data: For this project the Skin Cancer MNIST: HAM10000 Kaggle dataset was used. This contained about 10.000 moles images, as well some other files such a metadata file with following information:
   
- lesion_id
- image_id
- dx
- dx_type
- age
- sex
- localization

2. Clean, Prepare & Manipulate Data: A simple exploratory analysis took place to see the distribution of the data. Oversampling was done to increase the amount of certain mole types in the dataset, duplicates were removed and image augmentation.

3. Train Model: The Neural Network Classifier was trained using ImageNet from Tensorflow after performing a train, test and validation split.

4. Test Data: The model was tested with the test dataset.

5. Improve: the model is open for improvements in a further version, for instance hyperparameters tuning can be utilized to achieve better predictions.As well as to try other Tensorflow models and a better balancing of the data  . 


### Deployment

After the model was created and tested, the development of the Streamlit app took place and afterwards the deployment of the app on Heroku:


![deployment](https://miro.medium.com/max/1400/1*sPyUqSFLEXGYrezcq5LgTg.png)

### Visuals

The Streamlit application is a simple but functional solution where the user can first upload a picture of the mole:


### Uploading Mole image

![upload](https://github.com/sebasGarcia/challenge-mole/blob/dev/app_upload.JPG)



###  Obtaining Classification results

And afterwards, get the results of the mole classification on the screen.

![results](https://github.com/sebasGarcia/challenge-mole/blob/dev/app_results.JPG)

## Repo Architecture

```
|   .gitignore
|   .slugignore
|   app.py
|   app_results.JPG
|   app_upload.JPG
|   Model for Mole Dataset.ipynb
|   Procfile
|   README.md
|   requirements.txt
|   setup.sh
|   skincare.png
|
\---model
        my_model.h5  
```

## Installation

The following software, platforms and tools were utilised during the execution of the project:

* Python 3
* Anaconda Distribution
* Jupyter notebook
* Microsoft Excel
* Pandas library
* Scikit-learn library
* Tensorflow
* Matplotlib
* Seaborn
* Visual Studio Code
* Heroku
* Streamlit


## Usage

The application can be found through the following link:

https://skincare2app.herokuapp.com/

### Contributors:

Sebastián García martínez\
[![Linkedin](https://i.stack.imgur.com/gVE0j.png) https://www.linkedin.com/in/sebastiangarciamartinez](https://www.linkedin.com/in/sebastiangarciamartinez/)
&nbsp;



### Timeline:

8 days

16/05/2022 - 25/05/2022


![skincaree(GIF)](https://media.giphy.com/media/F1h0YEIl8fbLGuw6ny/giphy.gif)

