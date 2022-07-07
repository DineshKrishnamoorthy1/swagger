# Import Packages

import os
import cv2
import math
import hashlib
import time
import datetime
import uuid
import pytz
import pdfplumber
import numpy as np
import re
import string
from pathlib import Path
import matplotlib.pyplot as plt

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Classification Train

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

# Classification Test

from keras.preprocessing import image

import keras

import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

# paper itemization

from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter

# pdf to image

from pdf2image import convert_from_path

# Auto rotation

import pytesseract
from scipy.ndimage import rotate as Rotate

# Attribution
from agadia.Serializer import AgadiaSerializer
from kvp_attribution.core import getFileSize, getChecksum, do_kvp_attribution, form_recognizer
from kvp_attribution.invoicenet import FIELD_TYPES
from kvp_attribution.invoicenet.acp.acp import AttendCopyParse

#swagger
from rest_framework.generics import GenericAPIView



def getFileSize(filePath):
    file_size = os.path.getsize(filePath)
    size = str(round(file_size / (1024), 3))
    return size


def getFolderSize(filePath):
    original_folder_size = 0
    for path, dirs, files in os.walk(filePath):
        for f in files:
            fp = os.path.join(path, f)
            original_folder_size += os.path.getsize(fp)
    return str(round(original_folder_size / (1024 * 1024), 3)) + "MB"


def getChecksum(filePath):
    md5_hash = hashlib.md5()
    a_file = open(filePath, "rb")
    content = a_file.read()
    md5_hash.update(content)
    digest = md5_hash.hexdigest()
    return digest

# classification train

def classification_train(train_path,test_path,epochs,batch_size,img_width,img_height,output_path):

    image_size = [img_width,img_height]
    target_size = (img_width,img_height)

    classes = os.listdir(train_path)
    num_classes = len(classes)

    print("Class --> {} \n and the length is : {}".format(classes, num_classes))

    train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
    )
    training_set = train_datagen.flow_from_directory(
    directory = train_path,
    target_size = target_size,
    batch_size = batch_size,
    class_mode = 'categorical'
    )
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(
    directory = test_path,
    target_size = target_size,
    batch_size = batch_size,
    class_mode = 'categorical'
    )
    # Import the VGG 16 library as shown below and add preprocessing layer to the front of VGG
    # Here we will be using imagenet weights

    vgg = VGG16(input_shape = image_size + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False

    # our layers - you can add more if you want
    x = Flatten()(vgg.output)
    prediction = Dense(num_classes, activation='softmax')(x)

    # create a model object
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()

    # tell the model what cost and optimization method to use
    model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=epochs,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
    )
    # summarize history for loss
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation (Test) loss')
    plt.title('summarize history for loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('summarize history for accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #Save the vgg16 model
    model_name = "model.h5"
    model_path = output_path + model_name
    model.save(model_path)

    return model_path

# Paper itemization
def paper_itemization(input_file, output_path):

    localPdfPath = output_path + "/paper_itemization/"
    if not os.path.exists(localPdfPath):
        os.mkdir(localPdfPath)

    # Get a Filename and File Extension
    fileName, fileExtension = os.path.splitext(input_file)
    fileName = fileName.split('/')
    fileName = fileName[-1]

    fileNamePath = localPdfPath + str(fileName) + "/"
    if not os.path.exists(fileNamePath):
        os.mkdir(fileNamePath)

    inputpdf = PdfFileReader(open(input_file, "rb"))

    for i in range(inputpdf.numPages):
        output = PdfFileWriter()
        output.addPage(inputpdf.getPage(i))
        with open(fileNamePath + fileName + "_" + "%s.pdf" % i, "wb") as outputStream:
            output.write(outputStream)

    return fileNamePath

# Pdf to image
def pdf_to_image(file_path, output_path):

    image_conversion_file_path = output_path + "Image_conversion/"
    if not os.path.exists(image_conversion_file_path):
        os.mkdir(image_conversion_file_path)

    file_path_dir = file_path.split('/')
    file_name_path = file_path_dir[-2]

    image_conversion_file_path = image_conversion_file_path + file_name_path + "/"
    if not os.path.exists(image_conversion_file_path):
        os.mkdir(image_conversion_file_path)

    count = 0
    for i in os.listdir(file_path):

        # Get a Filename and File Extension
        file_name, file_extension = os.path.splitext(file_path + i)
        file_name = file_name.split('/')
        file_name = file_name[-1]

        # Store Pdf with convert_from_path function
        images = convert_from_path(file_path + i)
        for k in range(len(images)):
            count = count + 1

            # Save pages as images in the pdf
            images[k].save(image_conversion_file_path + str(file_name) + '.jpg', 'JPEG')

    return image_conversion_file_path


# Auto rotation Latest version

def auto_rotation(file_path, output_path):
    auto_rotaion_path = output_path + "auto_rotaion/"
    if not os.path.exists(auto_rotaion_path):
        os.mkdir(auto_rotaion_path)

    file_path_dir = file_path.split('/')
    file_name_path = file_path_dir[-2]

    auto_rotaion_path = auto_rotaion_path + file_name_path + "/"
    if not os.path.exists(auto_rotaion_path):
        os.mkdir(auto_rotaion_path)

    def float_convertor(x):
        if x.isdigit():
            out = float(x)
        else:
            out = x
        return out

    angles=[]
    for i in os.listdir(file_path):
        filename = file_path + '/' + i

        def tesseract_find_rotatation(img):
            img = cv2.imread(img) if isinstance(img, str) else img
            k = pytesseract.image_to_osd(img)
            out = {i.split(":")[0]: float_convertor(i.split(":")[-1].strip()) for i in k.rstrip().split("\n")}
            print(out)
            img_rotated = Rotate(img, 360 - out["Rotate"])
            cv2.imwrite(auto_rotaion_path + "/" + i, img_rotated)
            return out

        angle = tesseract_find_rotatation(filename)
        angles.append({i:angle})

    auto_rotaion_pdf_path = output_path + "auto_rotaion_pdf/"
    if not os.path.exists(auto_rotaion_pdf_path):
        os.mkdir(auto_rotaion_pdf_path)

    auto_rotaion_pdf_path = auto_rotaion_pdf_path + file_name_path + "/"
    if not os.path.exists(auto_rotaion_pdf_path):
        os.mkdir(auto_rotaion_pdf_path)

    for i in os.listdir(auto_rotaion_path):
        # Get a Filename and File Extension
        fileName, fileExtension = os.path.splitext(auto_rotaion_path + i)
        fileName = fileName.split('/')
        fileName = fileName[-1]

        img_read = Image.open(auto_rotaion_path + i).convert('RGB')
        img_read.save(auto_rotaion_pdf_path + fileName + ".pdf")

    return [auto_rotaion_path,auto_rotaion_pdf_path,angles]

# Document classification

def document_classification(input_file, model_name):

    target_size = (600, 600)
    img = image.load_img(input_file, target_size=target_size)
    Image.open(input_file)

    x = image.img_to_array(img)
    img = keras.utils.load_img(input_file, target_size=target_size)
    Image.open(input_file)

    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x_processed = preprocess_input(x)

    doc_model = load_model(model_name)

    classes = doc_model.predict(x_processed)
    classes = classes.round()

    if classes[0][0] == 1:
        return 'non-urgent'

    elif classes[0][1] == 1:
        return 'urgent'

    return None

# Text Detection and Recognition

def text_extraction(input_file, output_path):

    text_extract_pdf = []

    text_extraction_file_path = Path(output_path + "/text_extraction/")
    if not text_extraction_file_path.exists():
        text_extraction_file_path.mkdir(parents=True, exist_ok=True)

    file_name, file_extension = os.path.splitext(input_file)
    file_name = file_name.split('/')
    file_name = file_name[-1]

    file_name_path = str(text_extraction_file_path.absolute()) + "/" + str(file_name) + "/"
    if not os.path.exists(file_name_path):
        os.mkdir(file_name_path)

    pdf = pdfplumber.open(input_file)

    for page_no in range(len(pdf.pages)):
        page = pdf.pages[page_no]

        return page.extract_text()

    return None

# Attribution

def attribution(input_file, key, provided_fields):

    predictions = {}

    for field in key:
        model = AttendCopyParse(field=field, provided_fields=provided_fields)
        predict = model.predict(paths=[input_file], provided_fields=provided_fields)
        if len(predict) > 0:
            predictions[field] = predict[0]

    return predictions


# Checkbox attribution

def check_box_detection(input_path, output_path):

    checkbox_detection_path = output_path + "checkbox_detection/"
    if not os.path.exists(checkbox_detection_path):
        os.mkdir(checkbox_detection_path)

    checkbox_recognition_path = output_path + "checkbox_recognition/"
    if not os.path.exists(checkbox_recognition_path):
        os.mkdir(checkbox_recognition_path)

    checkbox_attribution_path = output_path + "checkbox_attribution/"
    if not os.path.exists(checkbox_attribution_path):
        os.mkdir(checkbox_attribution_path)

    fileName, fileExtension = os.path.splitext(input_path)
    fileName = fileName.split('/')
    fileName = fileName[-1]

    checkbox_recognition_path = checkbox_recognition_path + fileName + "/"
    if not os.path.exists(checkbox_recognition_path):
        os.mkdir(checkbox_recognition_path)

    checkbox_attribution_path = checkbox_attribution_path + fileName + "/"
    if not os.path.exists(checkbox_attribution_path):
        os.mkdir(checkbox_attribution_path)

    image = cv2.imread(input_path)

    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th1, img_bin = cv2.threshold(gray_scale, 150, 225, cv2.THRESH_BINARY)
    img_bin = ~img_bin

    line_min_width = 20

    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)

    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)

    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)

    img_bin_final = img_bin_h | img_bin_v

    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    def imshow_components(labels):
        label_hue = np.uint8(179 * labels / np.max(labels))
        empty_channel = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, empty_channel, empty_channel])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue == 0] = 0
        return labeled_img

    out_image = imshow_components(~labels)

    count = 0
    for x, y, w, h, area in stats[2:]:
        cv2.imwrite(f'{checkbox_recognition_path}{fileName}_{count}.jpg', image[y:y + h, x:x + w])

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(f'{checkbox_detection_path}/{fileName}.jpg', image)

        org_image = cv2.imread(input_path)
        cv2.imwrite(f'{checkbox_attribution_path}{fileName}_{count}.jpg',
                    org_image[y - 20:(y) + (h + 20), x:x + (w + 300)])

        count += 1

    output = []

    for i, j in zip(os.listdir(checkbox_recognition_path), os.listdir(checkbox_attribution_path)):

        text = pytesseract.image_to_string(Image.open(checkbox_attribution_path + j), lang='eng')
        text = text.replace("\n", " ")
        text = " ".join(re.findall("[a-zA-Z]+", text))

        if len(text) > 0:
            text = re.sub('[' + string.punctuation + ']', '', text).split()
            text_out = str(text[0] + " " + text[1])

            img = cv2.imread(checkbox_recognition_path + i, cv2.IMREAD_GRAYSCALE)

            n_white_pix = np.sum(img == 255)
            n_black_pix = np.sum(img == 0)

            if n_black_pix < 2:
                checkbox_out = "False"
            else:
                checkbox_out = "True"

            out_json = {text_out: checkbox_out}
            output.append(out_json)

    return output

parent_path = "/home/hwuser/Workspace/"

class Agadia(GenericAPIView):
    serializer_class = AgadiaSerializer
    def post (self,request,format=None):
        input_file = request.data.get('inputFilePath')
        model_path = request.data.get('modelFilePath')
        key = request.data.get('attributes')
        output_path = request.data.get('outputDir')

        input_file = parent_path + request.data.get('inputFilePath')
        model_path = parent_path + request.data.get('modelFilePath')
        key = request.data.get('attributes')
        output_path = parent_path + request.data.get('outputDir')

        # Get a Filename and File Extension
        file_name, file_extension = os.path.splitext(input_file)
        file_name = file_name.split('/')
        file_name = file_name[-1]
        file_extension = file_extension[1:]

        file_size = getFileSize(input_file)
        file_checksum = getChecksum(input_file)

        response = {}
        provided_fields = dict()
        for field in key:
            provided_fields[field] = FIELD_TYPES["general"]
        process_start = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
        start_sec = time.time()

        paper_itemization_output = paper_itemization(input_file, output_path)
        pdf_to_image_output = pdf_to_image(paper_itemization_output, output_path)
        auto_rotation_output = auto_rotation(pdf_to_image_output, output_path)

        auto_rotation_img_output = auto_rotation_output[0]
        auto_rotation_pdf_output = auto_rotation_output[1]

        model = "/home/ubuntu/workspace/agadia/data/models/classification_models/document_classification_model.h5"

        output = []

        for i in os.listdir(auto_rotation_img_output):
            document_classification_out = document_classification(auto_rotation_img_output + i, model)
            check_box_detection_out = check_box_detection(auto_rotation_img_output + i, output_path)

            # Get a Filename and File Extension
            file_name_, file_extension_ = os.path.splitext(auto_rotation_img_output + i)
            file_name_ = file_name_.split('/')
            file_name_ = file_name_[-1]
            file_extension_ = file_extension_[1:]

            res = {}

            text_stat_tuple = form_recognizer(auto_rotation_pdf_output + file_name_ + ".pdf", output_path)

            match = [u for u in key if str(u) in str(text_stat_tuple[7])]

            if len(match) >= 0:
                attribution_out = do_kvp_attribution(auto_rotation_pdf_output + file_name_ + ".pdf", key, model_path,
                                                     provided_fields)

            res['ItemizedPaperName'] = i
            res['documentType'] = document_classification_out
            res['extractedAttributes'] = attribution_out
            res['extractedText'] = text_stat_tuple[7]
            res['wordsCount'] = text_stat_tuple[0]
            res['wordsCountBelow60'] = text_stat_tuple[1]
            res['wordsCount60-80'] = text_stat_tuple[2]
            res['wordsCountAbove80'] = text_stat_tuple[3]

            res["documentClassBasedOnCheckbox"] = check_box_detection_out
            output.append(res)

        process_stop = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
        stop_sec = time.time()

        elapsed_time = round(stop_sec - start_sec, 2)

        response['status'] = status.HTTP_200_OK
        response['id'] = uuid.uuid1()

        response['inputFile'] = input_file
        response['inputFileName'] = file_name
        response['inputFileExtension'] = file_extension
        response['inputFileSizeInKb'] = file_size
        response['inputFileChecksum'] = file_checksum

        response['processStart'] = process_start
        response['processStop'] = process_stop
        response['elapsedTimeInSec'] = elapsed_time
        response['paperItemizationPath'] = paper_itemization_output
        response['imageConversionPath'] = paper_itemization_output
        response['autoRotationPath'] = auto_rotation_output[0]
        response['rotationInfo'] = auto_rotation_output[2]

        response['extractedInformation'] = output

        return Response(response)







