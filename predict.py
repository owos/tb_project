import numpy as np
import cv2 as cv
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pylot as plt
import os

# image laoder
# image laoder
def load_image(img_path, img_size, show=False):
    img = image.load_img(img_path, target_size=img_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # expanding image tensor
    img_tensor = img_tensor / 255.0  # scaling the image_T

    if show:
        plt.imshow(img_tensor[0])
        plt.axis("off")
        plt.show()
    return img_tensor


img_size = (300, 300)
img_path = "inference image from medscape.jpg"
model_path = "tb_model"


img_size = (300, 300)
img_path = "inference image from medscape.jpg"
model_path = "tb_model"
classes = ["Normal", "Tuberculosis"]
model = load_model(model_path)
if __name__ == "__main__":
    ## load img

    img = load_image(img_path)
    pred = model.predict()
    output = classes[round(pred[0][0])]
    st.write(f"The diagnosis is {output}")
