import streamlit
import tensorflow as tf
import os
import numpy as np
from details import disease_causes, disease_cures

streamlit.title('Plant disease detection')

labels = os.listdir('archive')

# remove '_' and '(' and ')' from labels and replace them with space
for i in range(len(labels)):
    labels[i] = labels[i].replace('_', ' ').title()


uploaded_file = streamlit.file_uploader("Choose an image...", type="jpg")

model = tf.keras.models.load_model('model_M_20.keras')


def predictions(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return labels[np.argmax(score)], 100 * tf.nn.sigmoid(score[np.argmax(score)]).numpy()


if uploaded_file is not None:
    streamlit.image(uploaded_file, caption='Uploaded Image.',
                    use_column_width=False)
    predictions, score = predictions(uploaded_file)
    streamlit.subheader(predictions)
    streamlit.write(
        f'This image most likely belongs to {predictions} with a {round(score, 2)} % confidence.')
    streamlit.write('Causes:')
    streamlit.markdown(disease_causes(predictions))
    streamlit.write('Cures:')
    streamlit.markdown(disease_cures(predictions))
