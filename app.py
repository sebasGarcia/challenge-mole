import streamlit as st
import tensorflow as tf
import json
import requests
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
from streamlit_lottie import st_lottie
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax


#Setting image icon for tab on Browser taken from <a href="https://www.flaticon.com/free-icons/skin" title="skin icons">Skin icons created by Victoruler - Flaticon</a>
icon_page = Image.open('skincare.png')

#Change default name of app on Browser Tab
st.set_page_config(page_title='SkinCare: Mole Detector App', page_icon=icon_page, layout="wide")
st.title("SkinCare: Mole Detector")

def main():
    """
    This main function contains the necessary code to run the Streamlit webapp
    """
    col1, col2 = st.columns([1,3])

    lottie_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_sxrzmxih.json")
    lottie_animation2 = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_zpjfsp1e.json")
    with st.container():
        with col1:
            st_lottie(lottie_animation,
            height=384,
            width=384)

        with col2:
            image_uploaded = st.file_uploader("Please upload the picture of the mole", type = ['jpg'])
            if image_uploaded is not None:
                image = Image.open(image_uploaded)
                       
                result = predict_class(image)
                st.subheader(result)
                st.image(image, use_column_width=False)

def predict_class(image):
    """
    This function receives an image and returns a prediction based on a saved model
    
    """
    cl_model = tf.keras.models.load_model("./model/my_model.h5")
    shape = ((224,224,3))
    model = tf.keras.Sequential([hub.KerasLayer(cl_model, input_shape=shape)])
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)

    classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}

    predictions = model.predict(test_image)
    print(predictions)
    #Showing more than 1 result. Ref https://stackoverflow.com/questions/43488194/get-order-of-class-labels-for-keras-predict-function

# sorting the predictions in descending order
    sorting = (-predictions).argsort()

# getting the top 4 predictions, can also be changed to all 7 or less
    sorted_ = sorting[0][:4]
    result = ""
    for value in sorted_:
 
        predicted_label = classes[value]

        prob = (predictions[0][value]) * 100
        prob = "%.2f" % round(prob,2)
        prob = str(prob) + "%"
      
        
        result += "{}The mole type is {} likely to belong to: {} - {} \n".format("\n",prob, predicted_label[0],predicted_label[1])

    return result

def load_lottieurl(url:str):
    """
    This function is used to show an animation on the webpage
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None 

    return r.json()

if __name__ == '__main__':
    main()

