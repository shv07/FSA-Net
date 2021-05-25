import cv2
import numpy as np
import base64

def base64_to_cv2_img(base64Image:str):
    """
    Converts base64 image to opencv readable image
    Usage:
        img = base64_to_cv2_img(image_b64)
        cv2.imshow(img)
    """

    nparr = np.fromstring(base64.b64decode(base64Image), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

