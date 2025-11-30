import numpy as np
from PIL import Image, ImageDraw

def mask_background(img, boxes):
    img_np = np.array(img)
    masked_np = np.zeros_like(img_np)
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        masked_np[y1:y2, x1:x2] = img_np[y1:y2, x1:x2]
    masked_img = Image.fromarray(masked_np)
    draw = ImageDraw.Draw(masked_img)
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    return masked_img