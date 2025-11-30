import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_image(image, window_name='Image', wait=True):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, image)
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)

def preview_dataset(dataset):
    for item in dataset:
        print(f"Label: {item['label']}")
        plt.imshow(item["image"])
        plt.title(item["label"])
        plt.axis('off')
        plt.show()