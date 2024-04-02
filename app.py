import streamlit
import tensorflow as tf
import os
import numpy as np
import io

streamlit.title('Plant disease detection')

labels = os.listdir('archive')

# remove '_' and '(' and ')' from labels and replace them with space
for i in range(len(labels)):
    labels[i] = labels[i].replace('_', ' ').title()
    labels[i] = labels[i].replace('(', ' ').title()
    labels[i] = labels[i].replace(')', ' ').title()


uploaded_file = streamlit.file_uploader("Choose an image...", type="jpg")

model = tf.keras.models.load_model('model_M_20.keras')


def predictions(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return labels[np.argmax(score)]


if uploaded_file is not None:
    streamlit.image(uploaded_file, caption='Uploaded Image.',
                    use_column_width=False)
    predictions = predictions(uploaded_file)
    streamlit.write(predictions)
