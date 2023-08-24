import tensorflow as tf
import numpy as np
from keras.preprocessing import image


model = tf.keras.models.load_model("model.keras")


test_image = image.load_img("dataset/single_prediction/4.jpg", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

if result[0][0] == 1:
    prediction = "Dog"
else:
    prediction = "Cat"

print(prediction)
