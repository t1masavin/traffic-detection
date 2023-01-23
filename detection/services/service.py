import cv2
import pandas as pd
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


def get_image(data, index):
    bbox_1 = data.iloc[index]['xmin'], data.iloc[index]['ymin'], data.iloc[index]['xmax'], data.iloc[index]['ymax']
    im = cv2.imread(data.iloc[index]["image"])
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im, bbox_1


def convertFromJson(data):
    data_list, data2_list = [], []
    data = data[data['region_count'] != 0]
    search_values = ['Name', 'Type']
    data = data[data.region_attributes.str.contains('|'.join(search_values ))]
    data = data.reset_index(drop=True)
    for row in range(data.shape[0]):
        try:
            data1 = eval(data['region_shape_attributes'][row])
            data2 = eval(data['region_attributes'][row])
            data_list.append(data1)
            data2_list.append(data2)
        except:
            data.drop([row], inplace=True)
            continue
    data = data.reset_index(drop=True)
    datadf = pd.DataFrame(data_list)
    datadf2 = pd.DataFrame(data2_list)
    print(datadf.columns)
    print(datadf2.columns)
    ##################################################################################################
    data = pd.concat([data, datadf, datadf2],1)[['filename','height','width','x','y', 'Name', 'Type']]
    ##################################################################################################
    # data = pd.concat([datadf, datadf2],1)[['filename','height','width','x','y', 'Name', 'Type']]
    data['Name'] = data['Name'].apply(lambda x: str(x).split('\n')[0])
    return data


def convertToXYmax(data):
    data_x2, data_y2 = [], []
    for row in range(data.shape[0]):
        data1 = data['x'][row] + data['width'][row]
        data2 = data['y'][row] + data['height'][row]
        data_x2.append(data1)
        data_y2.append(data2)
    datadf = pd.DataFrame(data_x2, columns=['xmax'])
    datadf2 = pd.DataFrame(data_y2, columns=['ymax'])
    data = pd.concat([data, datadf, datadf2],1)
    data = data.drop(columns=['height', 'width'])
    data.rename(columns={'filename': "image", 'x': "xmin", 'y': "ymin", 'Name': 'name', 'Type': 'class'}, inplace=True)
    data = data[['image', 'xmin', 'ymin', 'xmax', 'ymax', 'name', 'class']]
    return data
