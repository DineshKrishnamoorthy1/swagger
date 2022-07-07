import datetime
import json
import logging
import os
import time
import uuid

import cv2
import imutils as imutils
import numpy as np
import pytesseract
import pytz
from PIL import Image
from keras.models import load_model
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.generics import GenericAPIView
from sklearn.preprocessing import LabelBinarizer

from handwritten.Serializer import HandWrittenSerializer
from kvp_attribution.core import getFileSize, getChecksum

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger()


class HandwrittenAttribute(GenericAPIView):
    serializer_class = HandWrittenSerializer

    def post(self, request, format=None):
        input_file = request.data.get('inputFilePath')
        model_path = request.data.get('modelFilePath')
        output_path = request.data.get('outputDir')

        # Get a Filename and File Extension
        file_name, file_extension = os.path.splitext(input_file)
        file_name = file_name.split('/')
        file_name = file_name[-1]
        file_extension = file_extension[1:]

        file_size = getFileSize(input_file)
        file_checksum = getChecksum(input_file)

        response = {}

        process_start = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
        start_sec = time.time()

        # Create a table_denoise path
        table_denoise_path = output_path + "table_denoise/"
        if not os.path.exists(table_denoise_path):
            os.mkdir(table_denoise_path)

        # Get a Filename and File Extension
        file_name, file_extension = os.path.splitext(input_file)
        file_name = file_name.split('/')
        file_name = file_name[-1]

        table_denoise_path = table_denoise_path + file_name + "/"
        if not os.path.exists(table_denoise_path):
            os.mkdir(table_denoise_path)

        # Table denoise
        image = cv2.imread(input_file)
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal table lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

        # Remove vertical table lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

        # Saved table denoise file
        cv2.imwrite(f'{table_denoise_path}{file_name}.png', result)

        # Create a handwritten_detection path
        handwritten_detection_path = output_path + "handwritten_detection/"
        if not os.path.exists(handwritten_detection_path):
            os.mkdir(handwritten_detection_path)

        handwritten_detection_path = handwritten_detection_path + file_name + "/"
        if not os.path.exists(handwritten_detection_path):
            os.mkdir(handwritten_detection_path)

        image = cv2.imread(f'{table_denoise_path}{file_name}.png')

        # detection of hand written contents with keys
        count = 0
        values = [[100, 0, 105, 600], [550, 0, 100, 320], [550, 320, 100, 370], [550, 680, 100, 870],
                  [100, 580, 105, 870],
                  [350, 0, 120, 870]]
        for i in values:
            crop = image[i[0]:i[0] + i[2], i[1]:i[1] + i[3]]
            cv2.imwrite(f'{handwritten_detection_path}{file_name}_{count}.png', crop)
            count += 1

        # Create a handwritten_detection_crop path
        handwritten_detection_crop_path = output_path + "handwritten_detection_crop/"
        if not os.path.exists(handwritten_detection_crop_path):
            os.mkdir(handwritten_detection_crop_path)

        handwritten_detection_crop_path = handwritten_detection_crop_path + file_name + "/"
        if not os.path.exists(handwritten_detection_crop_path):
            os.mkdir(handwritten_detection_crop_path)

        # Handwritten recognition

        # Load model
        l = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
             'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        model = load_model(model_path)

        def sort_contours(cnts, method="left-to-right"):
            reverse = False
            i = 0
            if method == "right-to-left" or method == "bottom-to-top":
                reverse = True
            if method == "top-to-bottom" or method == "bottom-to-top":
                i = 1
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                key=lambda b: b[1][i], reverse=reverse))
            return (cnts, boundingBoxes)

        def get_letters(img):
            letters = []
            image = cv2.imread(img)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            dilated = cv2.dilate(thresh1, None, iterations=2)

            cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sort_contours(cnts, method="left-to-right")[0]
            for c in cnts:
                if cv2.contourArea(c) > 10:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
                thresh = thresh.astype("float32") / 255.0
                thresh = np.expand_dims(thresh, axis=-1)
                thresh = thresh.reshape(1, 32, 32, 1)
                ypred = model.predict(thresh)
                LB = LabelBinarizer()

                LB.fit_transform(l)
                ypred = LB.inverse_transform(ypred)
                [x] = ypred
                letters.append(x)
            return letters, image

        def get_word(letter):
            word = "".join(letter)
            return word

        def text_extraction(input_file):
            img = Image.open(input_file)
            img.load()
            text = pytesseract.image_to_string(img, lang="eng")
            return text

        count = 0
        output = {}
        for i in os.listdir(handwritten_detection_path):

            im = Image.open(handwritten_detection_path + i)

            # Get dimensions
            width, height = im.size

            left_upper = 0
            top_upper = 0
            right_upper = width
            bottom_upper = height / 2

            # Crop the center of the image
            im_upper = im.crop((left_upper, top_upper, right_upper, bottom_upper))
            im_upper.save(f"{handwritten_detection_crop_path}{file_name}_upper_{count}.png")
            extracted_text = text_extraction(f"{handwritten_detection_crop_path}{file_name}_upper_{count}.png")
            print("Key                :", extracted_text)

            left_lower = 0
            top_lower = height / 2
            right_lower = width
            bottom_lower = height

            # Crop the center of the image
            im_lower = im.crop((left_lower, top_lower, right_lower, bottom_lower))
            im_lower.save(f"{handwritten_detection_crop_path}{file_name}_lower_{count}.png")

            letter, image = get_letters(f"{handwritten_detection_crop_path}{file_name}_lower_{count}.png")
            word = get_word(letter)
            print("Values              :", word)
            extracted_text_ = extracted_text.replace("\n\f", "")
            output[extracted_text_] = word

            print("     ")
            if str(extracted_text) == "Date of Birth (dd/mm/yyyy)\n\x0c":
                print("(dd/mm/yyyy)        -", word[0:2] + "/" + word[2:4] + "/" + word[4:8])

            if str(extracted_text) == "FULL NAME, in CAPITAL Letters (In the order of Title (Mr./Mrs./etc.first, middle, and last name, leaving a space between words))\n\x0c":
                print("Mr./Mrs./etc       -", word[0:2])
                print("First Name         -", word[2:7])
                print("Last Name          -", word[7::])

            count += 1

            print("----------------------------------------------------------------------------------")

        process_stop = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
        stop_sec = time.time()

        elapsed_time = round(stop_sec - start_sec, 2)

        response['status'] = status.HTTP_200_OK
        response['id'] = str(uuid.uuid1())

        response['inputFile'] = input_file
        response['inputFileName'] = file_name
        response['inputFileExtension'] = file_extension
        response['inputFileSizeInKb'] = file_size
        response['inputFileChecksum'] = file_checksum

        response['processStart'] = process_start
        response['processStop'] = process_stop
        response['elapsedTimeInSec'] = elapsed_time
        response['highlightedAttribute'] = output

        json_response_Path = output_path + "json_response/"
        if not os.path.exists(json_response_Path):
            os.mkdir(json_response_Path)

        json_object = json.dumps(response, indent=4)
        with open(f"{json_response_Path}/{file_name}.json", 'w') as f:
            f.write(json_object)

        logger.info(response)

        return Response(response)
