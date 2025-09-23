import cv2
import numpy as np


def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
    return img

def resize_for_display(image, max_height=500):
    h, w = image.shape
    if h > max_height:
        scale = max_height / w
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

def overlay_edges(image, edges, color=(255, 0, 0)):
    # Chuẩn hoá edges về uint8 [0, 255]
    if edges.dtype != np.uint8:
        edges = np.clip(edges, 0, 255).astype(np.uint8)
    else:
        edges = edges.copy()

    # Nếu edges không phải nhị phân (0/255) → threshold
    if len(np.unique(edges)) > 2:  
        _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

     
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Tạo mask từ edges
    mask = edges > 0
    image_rgb[mask] = color

    return image_rgb