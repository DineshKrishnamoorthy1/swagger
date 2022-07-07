import datetime
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path

import camelot
import fitz
import numpy as np
import pdfplumber
import pytz
from PIL import Image
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.generics import GenericAPIView

from asset_attribution_model.Serializer import doAssetAttributionSerializer
from attribution_train.AttributionTrainSchema import AttributionTrainSchema
from kvp_attribution.core import getFileSize, getChecksum, do_kvp_attribution, form_recognizer
from kvp_attribution.invoicenet import FIELD_TYPES
from kvp_attribution.invoicenet.acp.acp import AttendCopyParse

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger()

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import hashlib


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


# paper_itemization

def paper_itemization(input_file, output_path):
    logging.info("Paper Itemization Started")

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


# Multi lines out table Extraction

def multi_lines_out_table_extract(input_file, output_path):
    logger.info("Multi line out Table Extraction Started")

    # Create a tableExtract path
    tableExtractPath = output_path + "/multiline-item_table_extract/"
    if not os.path.exists(tableExtractPath):
        os.mkdir(tableExtractPath)

    # Get a Filename and File Extension
    fileName, fileExtension = os.path.splitext(input_file)
    fileName = fileName.split('/')
    fileName = fileName[-1]

    # Create a tableExtract path
    filenamePath = tableExtractPath + fileName + "/"
    if not os.path.exists(filenamePath):
        os.mkdir(filenamePath)

    table_regions = ["0,800,600,0"]

    tables = camelot.read_pdf(input_file, flavor='stream', pages='all', table_regions=table_regions, row_tol=10,
                              column_tol=0)

    tableCount = len(tables)
    list_json = []
    for tabcounts in range(tableCount):
        df = tables[tabcounts].df

        df = df[1:]

        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header

        report = tables[tabcounts].parsing_report

        page = report.get("page")
        order = report.get("order")

        df.to_csv(filenamePath + fileName + "_" + str(page) + "_" + str(order) + ".csv")
        json_out = df.to_json(orient='split')
        json_out = json.loads(json_out)
        list_json.append(json_out)

    return list_json


# Text Detection and Recognition


# Text Detection and Recognition

def text_extraction(input_file, output_path):
    text_extract_pdf = []

    text_extraction_file_path = Path(output_path + "/text_extraction/")
    if not text_extraction_file_path.exists():
        text_extraction_file_path.mkdir(parents=True, exist_ok=True)

    # Get a Filename and File Extension
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


# Asset Extraction

def asset_extraction(input_file, output_path):
    logger.info("Asset Extraction Started")

    # Create a asset extract path
    imageExtractPath = output_path + "/asset_extraction/"
    if not os.path.exists(imageExtractPath):
        os.mkdir(imageExtractPath)

    # Get a Filename and File Extension
    fileName, fileExtension = os.path.splitext(input_file)
    fileName = fileName.split('/')
    fileName = fileName[-1]

    fileNamePath = imageExtractPath + fileName + "/"
    if not os.path.exists(fileNamePath):
        os.mkdir(fileNamePath)

    # open file
    with fitz.open(input_file) as my_pdf_file:

        output_files = []
        # loop through every page
        for page_number in range(1, len(my_pdf_file) + 1):

            # access individual page
            page = my_pdf_file[page_number - 1]

            # accesses all images of the page
            images = page.getImageList()

            # check if images are there
            if images:
                logger.info(f"There are {len(images)} image/s on page number {page_number}[+]")
            else:
                logger.info(f"There are No image/s on page number {page_number}[!]")

            # loop through all images present in the page

            for image_number, image in enumerate(page.getImageList(), start=1):
                # access image xerf
                xref_value = image[0]

                # extract image information
                base_image = my_pdf_file.extractImage(xref_value)

                # access the image itself
                image_bytes = base_image["image"]

                # get image extension
                ext = base_image["ext"]

                # load image
                image = Image.open(io.BytesIO(image_bytes))

                # save image locally

                output_file_name = f"{fileNamePath}{fileName}_{page_number}_Image{image_number}.{ext}"

                image.save(open(output_file_name, "wb"))

                output_files.append(output_file_name)

    return [fileNamePath, output_files]


# asset Classification

def asset_classification(model_path, file_name, asset_extraction_output_path, output_path, provided_fields=None):
    logger.info("Asset Classification Started")

    asset_extraction_output_path = str(asset_extraction_output_path)

    image_classify_folder = output_path + "/asset_classification/"
    if not os.path.exists(image_classify_folder):
        os.mkdir(image_classify_folder)

    image_classify_folder = image_classify_folder + str(file_name)
    if not os.path.exists(image_classify_folder):
        os.mkdir(image_classify_folder)

    certificate = image_classify_folder + "/CERTIFIED"
    if not os.path.exists(certificate):
        os.mkdir(certificate)

    graph = image_classify_folder + "/GRAPH"
    if not os.path.exists(graph):
        os.mkdir(graph)

    industry = image_classify_folder + "/IMAGE"
    if not os.path.exists(industry):
        os.mkdir(industry)

    logo = image_classify_folder + "/LOGO"
    if not os.path.exists(logo):
        os.mkdir(logo)

    maps = image_classify_folder + "/GEO"
    if not os.path.exists(maps):
        os.mkdir(maps)

    signature = image_classify_folder + "/SIGNATURE"
    if not os.path.exists(signature):
        os.mkdir(signature)

    table = image_classify_folder + "/MULTI_LINE_ITEM_TABLE"
    if not os.path.exists(table):
        os.mkdir(table)

    predictions = []

    for input_file in os.listdir(asset_extraction_output_path):

        input_file = str(input_file)

        target_size = (600, 600)
        img = image.load_img(asset_extraction_output_path + input_file, target_size=target_size)
        im = Image.open(asset_extraction_output_path + input_file)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x_processed = preprocess_input(x)

        # Load model
        from keras.models import load_model
        new_model = load_model(model_path)

        classes = new_model.predict(x_processed)

        classes = classes.round()

        if classes[0][0] == 1:
            prediction = 'certificate'
            predictions.append(prediction)
            im.save(certificate + "/" + input_file)

        elif classes[0][1] == 1:
            prediction = 'graph'
            predictions.append(prediction)
            im.save(graph + "/" + input_file)

        elif classes[0][2] == 1:
            prediction = 'industry'
            predictions.append(prediction)
            im.save(industry + "/" + input_file)

        if classes[0][3] == 1:
            prediction = 'logo'
            predictions.append(prediction)
            im.save(logo + "/" + input_file)

        elif classes[0][4] == 1:
            prediction = 'maps'
            predictions.append(prediction)
            im.save(maps + "/" + input_file)

        elif classes[0][5] == 1:
            prediction = 'signature'
            predictions.append(prediction)
            im.save(signature + "/" + input_file)

        elif classes[0][6] == 1:
            prediction = 'table'
            predictions.append(prediction)
            im.save(table + "/" + input_file)

    assets_classified_path = {}

    path = [x[0] for x in os.walk(image_classify_folder)]
    path.pop(0)

    for files in path:

        sub_dir = files.split('/')
        sub_dir = sub_dir[-1]

        predicted_filenames = [(files.replace(output_path, "") + "/" + item) for item in os.listdir(files)]

        if len(predicted_filenames) > 0:
            assets_classified_path[sub_dir] = predicted_filenames

    return assets_classified_path


def attribution(input_file, key, provided_fields):
    logger.info("Attribution Started")

    predictions = {}

    for field in key:
        model = AttendCopyParse(field=field, provided_fields=provided_fields)
        predict = model.predict(paths=[input_file], provided_fields=provided_fields)
        if len(predict) > 0:
            predictions[field] = predict[0]

    return predictions


class DoAssetAttribution(GenericAPIView):
    serializer_class = doAssetAttributionSerializer
    def post(self, request, format=None):
       
        input_file = request.data.get('inputFilePath')
        model_path = request.data.get('modelFilePath')
        model_type = request.data.get('modelType')
        key = request.data.get('attributes')
        output_path = request.data.get('outputDir')

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

        if model_path is not None and model_type == 'ASSET_EXTRACTION':
            asset_extraction_output = asset_extraction(input_file, output_path)
            response['classifiedAssets'] = asset_classification(model_path, file_name, asset_extraction_output[0],
                                                                output_path, provided_fields)
            # multi_lines_out_table_extract_output = multi_lines_out_table_extract(input_file, output_path)
            # response['tables'] = multi_lines_out_table_extract_output
            response['extractedAssetFilePaths'] = asset_extraction_output[1]

        if key is not None and model_path is not None and model_type == 'ASSET_ATTRIBUTION':

            if file_extension == 'pdf':
                response['extractedText'] = text_extraction(input_file, output_path)

            extracted_content = response['extractedText']

            # print(extracted_content)

            if extracted_content == '':
                text_stat_tuple = form_recognizer(input_file, output_path)
                response['extractedText'] = text_stat_tuple[7]
                response['words_count'] = text_stat_tuple[0]
                response['words_count_Below60'] = text_stat_tuple[1]
                response['words_count_60-80'] = text_stat_tuple[2]
                response['words_count_Above 80'] = text_stat_tuple[3]

            text = response['extractedText']
            match = [u for u in key if str(u) in str(text)]
            if len(match) >= 0:
                attribution_output = do_kvp_attribution(input_file, match, model_path, provided_fields)
                response['highlightedAttribute'] = attribution_output

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

        logger.info(response)
        return Response(response)
