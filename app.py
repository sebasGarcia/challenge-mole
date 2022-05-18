import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py


st.header("SkinCare: Mole Detector")

def main():
    image_uploaded = st.file_uploader("Please upload the picture of the mole", type = ['jpg'])
    if image_uploaded is not None:
        image = Image.open(image_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        #here I will have to put the name of the class that predicts:
        result = predict_class(image)
        st.subheader(result)
        st.pyplot(figure)

def predict_class(image):
    cl_model = tf.keras.models.load_model("model\my_model.h5")
    shape = ((224,224,3))
    model = tf.keras.Sequential([hub.KerasLayer(cl_model, input_shape=shape)])
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)

    classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}

    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])

    scores = scores.numpy()
    image_type = classes[np.argmax(scores)]
    print(image_type)
    
    result = "The mole type is likely: {} - {}".format(image_type[0], image_type[1])

    return result

if __name__ == '__main__':
    main()

