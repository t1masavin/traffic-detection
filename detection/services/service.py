import cv2
import matplotlib.pyplot as plt

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    bbox = [int(i) for i in bbox]
    x_min, y_min, x_max, y_max = bbox

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(
        str(class_name), cv2.FONT_HERSHEY_SIMPLEX, 0.1, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.5 * text_height)),
                  (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=str(class_name),
        org=(x_min- int(2 * text_width), y_min - int(0.5 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=color,
        thickness= 1
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name=None, category_id_to_color=None):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        color_name =category_id_to_color[category_id]
        img = visualize_bbox(img, bbox, class_name, color_name)
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.imshow(img, aspect='auto')
