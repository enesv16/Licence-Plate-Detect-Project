import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
import pytesseract
import datetime
import asyncio
from core.config import cfg
import re
from PIL import Image
import os



# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

import firebase_admin
from firebase_admin import credentials, firestore, db, storage


# Use the private key file of the service account directly.
cred = credentials.Certificate("./service_account_key.json")
fb_app = firebase_admin.initialize_app(cred, {'databaseURL': 'https://bitirme-8af0e-default-rtdb.firebaseio.com/', 'storageBucket': 'bitirme-8af0e.appspot.com'})
firestore_db = firestore.client()


vehicles_ref = firestore_db.collection('Vehicles')
vehicle_docs = vehicles_ref.stream()

vehicles = []
for vehicle_doc in vehicle_docs:
    license_plate = vehicle_doc.get('LicensePlate')
    is_wanted = vehicle_doc.get('IsWanted')
    vehicles.append({'license_plate': license_plate, 'is_wanted': is_wanted})
print(vehicles)

bucket = storage.bucket()

realtime_alert_ref = db.reference('AlertInfo')

async def get_stored_vehicle(license_plate):
    for vehicle in vehicles:
        if vehicle['license_plate'] == license_plate:
            return vehicle
    return None

async def realtime_alert(license_plate, img_data):
    print("REALTIME ALERT STARTED")
    current_datetime = datetime.datetime.now()
    img = Image.fromarray(img_data, 'RGB')
    img_filename = f'{license_plate}_{current_datetime.strftime("%Y-%m-%d-%H-%M")}.png'
    img_relative_filepath = f'detections\\wanted_imgs\\{img_filename}'
    img_absolute_filepath = os.path.join(os.getcwd(), img_relative_filepath)
    img.save(img_absolute_filepath)
    blob = bucket.blob(img_filename)
    blob.upload_from_filename(img_absolute_filepath)
    blob.make_public()
    img_url = blob.public_url

    realtime_alert_ref.set({
        'LicensePlate': license_plate,
        'Date': current_datetime.isoformat(),
        'Location': 'P8VQ+HW Serdivan, Sakarya',
        'ImageUrl': img_url
    })

    print("REALTIME ALERT ENDED")
    #upload the image to firebase storage

# function to recognize license plate numbers using Tesseract OCR
async def recognize_plate(img, coords):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    # grayscale region within bounding box
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu Threshold", thresh)
    #cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    #cv2.imshow("Dilation", dilation)
    #cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    
    text = pytesseract.image_to_string(blur, lang ='eng', config ='-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" --oem 3 --psm 6')
    clean_text = re.sub('[\W_]+', '', text)
    plate_num = clean_text

    # if plate_num != None:      
    #     if len(plate_num)>5:
    #         # sonuc=imlec.execute('SELECT * FROM araclar WHERE plaka=? AND ceza=?',(plate_num,0))
    #         # araclar = imlec.fetchall()
    #         araclar = vehicles_ref.where(u'plate', u'==', plate_num).where(u'penalty', u'==', 0).stream();
    #         if len(araclar)>0:
    #             print("Cezali License Plate #: ", plate_num)
    #             return "CEZALI ->" + plate_num
    #         else:
    #             print("License Plate #: ", plate_num)
    is_wanted = False
    if plate_num != None:
        if len(plate_num)>5:
            stored_vehicle = await get_stored_vehicle(plate_num)
            # License plate is found on the database
            if stored_vehicle != None:
                is_wanted = stored_vehicle['is_wanted']
                if (is_wanted):
                    print("Cezali License Plate #: ", plate_num)
                    await realtime_alert(plate_num, img)
            else:
                print("License Plate #: ", plate_num)
            # for vehicle in vehicles:
            #     stored_license_plate = vehicle['license_plate']
            #     stored_is_wanted = vehicle['is_wanted']
            #     if stored_license_plate == plate_num:
            #         is_wanted = stored_is_wanted
            #         if (is_wanted):
            #             print("Cezali License Plate #: ", plate_num)
            #             await realtime_alert(plate_num, img)
            #             break
            #     else:
            #         print("License Plate #: ", plate_num)
            #         break
            # if any(plate_num == vehicle.get('license_plate') for vehicle in vehicles):
    return plate_num, is_wanted


def load_weights(model, weights_file, model_name='yolov4'):

    layer_size = 110
    output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_config(FLAGS):
    STRIDES = np.array(cfg.YOLO.STRIDES)
    if FLAGS.model == 'yolov4':
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS)
    XYSCALE = cfg.YOLO.XYSCALE if FLAGS.model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def get_anchors(anchors_path):
    anchors = np.array(anchors_path)
    return anchors.reshape(3, 3, 2)

# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes

async def draw_bbox(image, bboxes, counted_classes = None, show_label=True, allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values()), read_plate = False):
    classes = read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]
        if class_name not in allowed_classes:
            continue
        else:
            if read_plate:
                height_ratio = int(image_h / 25)
                plate_number, is_wanted = await recognize_plate(image, coor)
                if plate_number != None:
                    if is_wanted:
                        cv2.putText(image, f'CEZALI -> {plate_number}', (int(coor[0]), int(coor[1]-height_ratio)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,0,0), 2)
                    else:
                        cv2.putText(image, plate_number, (int(coor[0]), int(coor[1]-height_ratio)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.25, (17,18,254), 2)

            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.int32(c3[0]), np.int32(c3[1])), bbox_color, -1) #filled

                cv2.putText(image, bbox_mess, (c1[0], np.int32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

            if counted_classes != None:
                height_ratio = int(image_h / 25)
                offset = 15
                for key, value in counted_classes.items():
                    cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    offset += height_ratio
    return image