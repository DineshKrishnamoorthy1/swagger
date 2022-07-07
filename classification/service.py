import datetime
import logging
import os
import time
import uuid

import numpy as np
import pytz
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from pdf2image import convert_from_path
from rest_framework import status
from rest_framework.decorators import api_view

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

logger = logging.getLogger(__name__)

KOLKATA_ZONE = 'Asia/Kolkata'
MODEL_PATH = 'modelPath'
OUTPUT_DIR = 'outputDir'
INPUT_FILE_PATH = 'inputFilePath'


@api_view(['POST'])
def do_classification(request):
    process_start = datetime.datetime.now(pytz.timezone(KOLKATA_ZONE)).strftime(TIME_FORMAT)
    start_sec = time.time()
    logger.info('--------------------------------------------------------')
    logger.info(request)

    input_file = request.data.get(INPUT_FILE_PATH)
    output_dir = request.data.get(OUTPUT_DIR)
    model_path = request.data.get(MODEL_PATH)

    response = {'id': str(uuid.uuid1())}
    try:
        document_classification_ = document_classification(pdftoimage_output + i, countryModelName)

        response['status'] = status.HTTP_200_OK

    except Exception as e:
        response['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
        logger.error(e)

    process_stop = datetime.datetime.now(pytz.timezone(KOLKATA_ZONE)).strftime(TIME_FORMAT)
    stop_sec = time.time()

    response['processStart'] = process_start
    response['processStop'] = process_stop
    response['elapsedTimeInSec'] = round(stop_sec - start_sec, 2)

    logger.info(response)
    return response


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


# Document classification
def document_classification(file_path, model_name):
    target_size = (600, 600)
    img = image.load_img(file_path, target_size=target_size)
    Image.open(file_path)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x_processed = preprocess_input(x)

    from keras.models import load_model
    new_model = load_model(model_name)

    classes = new_model.predict(x_processed)
    classes = classes.round()

    if classes[0][0] == 1:
        return 'AU'

    elif classes[0][1] == 1:
        return 'BR'

    elif classes[0][2] == 1:
        return 'CN'

    elif classes[0][3] == 1:
        return 'ES'

    elif classes[0][4] == 1:
        return 'IT'

    elif classes[0][5] == 1:
        return 'JP'

    elif classes[0][6] == 1:
        return 'KR'

    elif classes[0][7] == 1:
        return 'US'

    return None
