# Create your tests here.

import datetime
import glob
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time

import pandas as pd
import pytz
import simplejson
import tqdm
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pdf2image import pdf2image

from kvp_attribution.invoicenet import FIELD_TYPES
from kvp_attribution.invoicenet.acp.acp import AttendCopyParse
from kvp_attribution.invoicenet.acp.data import InvoiceData
from kvp_attribution.invoicenet.common import trainer
from kvp_attribution.invoicenet.common import util


# output_path = "halo_asis/output/"


def getFileSize(filePath):
    file_size = os.path.getsize(filePath)
    size = str(round(file_size / (1024), 3))
    return size


def getChecksum(filePath):
    md5_hash = hashlib.md5()
    a_file = open(filePath, "rb")
    content = a_file.read()
    md5_hash.update(content)
    digest = md5_hash.hexdigest()
    return digest


# online attribution

def do_kvp_attribution(input_file, key, model_path, provided_fields=None):
    predictions = {}
    for field in key:
        attribution(field, input_file, model_path, predictions, provided_fields)
    return predictions


def attribution(field, input_file, model_path, predictions, provided_fields=None):
    if provided_fields is None:
        provided_fields = dict()
    model = AttendCopyParse(field=field, restore=True, model_path=model_path, provided_fields=provided_fields)
    predict = model.predict(paths=[input_file], provided_fields=provided_fields)
    if len(predict) > 0:
        predictions[field] = predict[0]


# Text Detection and Recognition

def form_recognizer(file_path, output_path):
    # Creating a Output Folders

    text_recognize_json_path = output_path + "/TextRecognizeJSON2CSVPath"
    if not os.path.exists(text_recognize_json_path):
        os.mkdir(text_recognize_json_path)

    # Text Detection and Recognition Model Import
    ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    model = ocr_predictor(pretrained=True)

    # Get a Filename and File Extension
    file_name, file_extension = os.path.splitext(file_path)
    file_name = file_name.split('/')
    file_name = file_name[-1]

    doc = None

    processStart = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    startSec = time.time()

    # Input Pdf
    if file_extension == '.pdf':
        doc = DocumentFile.from_pdf(file_path).as_images()
        # words = DocumentFile.from_pdf(filePath).get_words()

    # Input Image
    elif file_extension == '.jpg':
        doc = DocumentFile.from_images(file_path)

    # Text Detection
    output = model(doc)

    # Detected Text Visualization
    # output.show(doc)

    # Json Export
    json_output = output.export()

    processStop = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    stopSec = time.time()

    elapsedTime = stopSec - startSec

    # Extracting the words,confidence score and Geometry
    out = []
    for words in json_output["pages"][0]["blocks"][0]["lines"][0]["words"]:
        out.append(words)

    # Output DataFrame
    df = pd.DataFrame(out)
    df.to_csv(text_recognize_json_path + "/" + file_name + ".csv", index=False)

    values = df['value']
    values = list(values)

    paraGraph = '  '.join(values)

    teA60 = df['confidence'][(df['confidence'] < 0.60)].count()
    teA6080 = df['confidence'][(df['confidence'] >= 0.60) & (df['confidence'] <= 0.80)].count()
    teA80 = df['confidence'][(df['confidence'] > 0.80)].count()

    wordsCount = df['value'].count()

    output = [wordsCount, teA60, teA6080, teA80, processStart, processStop, elapsedTime, paraGraph]

    return output


def attribution_train(model_data_abs_path, processed_data_abs_path, field, batchSize, restore, steps, earlyStopSteps,
                      provided_fields):
    train_data = InvoiceData.create_dataset(field=field,
                                            data_dir=os.path.join(processed_data_abs_path, 'train/'),
                                            batch_size=batchSize, provided_fields=provided_fields)

    val_data = InvoiceData.create_dataset(field=field,
                                          data_dir=os.path.join(processed_data_abs_path, 'val/'),
                                          batch_size=batchSize, provided_fields=provided_fields)

    print("Training...")

    trainer.train(
        model=AttendCopyParse(field=field, model_path=model_data_abs_path, provided_fields=provided_fields),
        train_data=train_data,
        val_data=val_data,
        total_steps=steps,
        early_stop_steps=earlyStopSteps
    )

    return "Success"


# def fileGenerator(inputFile, key, value, out_dir):
#     fileName, fileExtension = os.path.splitext(inputFile)
#     fileName = fileName.split('/')
#     fileName = fileName[-1]
#
#     def VersionFile(source, destination):
#
#         name, extension = os.path.splitext(destination)
#         for i in range(5):
#             new_file = f'{name} ({i}){extension}'
#             if not os.path.isfile(new_file):
#                 shutil.copy(source, new_file)
#                 continue
#
#     # Data to be written
#     attributeData = {
#
#         key: value
#     }
#
#     # Serializing json
#     json_object = json.dumps(attributeData, indent=4)
#
#     # Writing to sample.json
#
#     shutil.copy(inputFile, out_dir)
#
#     with open(out_dir + fileName + ".json", "w") as outfile:
#         outfile.write(json_object)
#
#     pdf_file = f'{out_dir}{fileName}.pdf'
#     json_file = f'{out_dir}{fileName}.json'

    # pdf_copy_file = VersionFile(pdf_file, pdf_file)
    # json_copy_file = VersionFile(json_file, json_file)

def fileGenerator(input_files, key, values, out_dir):
    for input_file, value in zip(input_files, values):

        fileName, fileExtension = os.path.splitext(input_file)
        fileName = fileName.split('/')
        fileName = fileName[-1]

        def VersionFile(source, destination):

            name, extension = os.path.splitext(destination)
            for i in range(5):
                new_file = f'{name} ({i}){extension}'
                if not os.path.isfile(new_file):
                    shutil.copy(source, new_file)
                    continue

        # Data to be written
        attributeData = {

            key: value
        }

        # Serializing json
        json_object = json.dumps(attributeData, indent=4)

        # Writing to sample.json

        shutil.copy(input_file, out_dir)

        with open(out_dir + fileName + ".json", "w") as outfile:
            outfile.write(json_object)

        pdf_file = f'{out_dir}{fileName}.pdf'
        json_file = f'{out_dir}{fileName}.json'

        pdf_copy_file = VersionFile(pdf_file, pdf_file)
        json_copy_file = VersionFile(json_file, json_file)

def prepare_data(train_data_abs_path, processed_data_abs_path, provided_fields):
    val_size = 0.2
    cores = max(1, (mp.cpu_count() - 2) // 2)
    ocr_engine = "pytesseract"

    os.makedirs(os.path.join(processed_data_abs_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_abs_path, 'val'), exist_ok=True)

    filenames = [os.path.abspath(f) for f in glob.glob(train_data_abs_path + "**/*.pdf", recursive=True)]

    idx = int(len(filenames) * val_size)
    train_files = filenames[idx:]
    val_files = filenames[:idx]

    print("Total: {}".format(len(filenames)))
    print("Training: {}".format(len(train_files)))
    print("Validation: {}".format(len(val_files)))

    for phase, filenames in [('train', train_files), ('val', val_files)]:
        print("Preparing {} data...".format(phase))

        with tqdm.tqdm(total=len(filenames)) as pbar:
            pool = mp.Pool(cores)
            for filename in filenames:
                pool.apply_async(process_file,
                                 args=(filename, processed_data_abs_path, phase, ocr_engine, provided_fields),
                                 callback=lambda _: pbar.update())

            pool.close()
            pool.join()


def process_file(filename, out_dir, phase, ocr_engine, provided_fields):
    try:
        page = pdf2image.convert_from_path(filename)[0]
        page.save(os.path.join(out_dir, phase, os.path.basename(filename)[:-3] + 'png'))

        height = page.size[1]
        width = page.size[0]

        ngrams = util.create_ngrams(page, height=height, width=width, ocr_engine=ocr_engine)
        for ngram in ngrams:
            if "amount" in ngram["parses"]:
                ngram["parses"]["amount"] = util.normalize(ngram["parses"]["amount"], key="amount")
            if "date" in ngram["parses"]:
                ngram["parses"]["date"] = util.normalize(ngram["parses"]["date"], key="date")

        with open(filename[:-3] + 'json', 'r') as fp:
            labels = simplejson.loads(fp.read())

        fields = {}
        for field in provided_fields:
            if field in labels:
                if provided_fields[field] == FIELD_TYPES["amount"]:
                    fields[field] = util.normalize(labels[field], key="amount")
                elif provided_fields[field] == FIELD_TYPES["date"]:
                    fields[field] = util.normalize(labels[field], key="date")
                else:
                    fields[field] = labels[field]
            else:
                fields[field] = ''

        data = {
            "fields": fields,
            "nGrams": ngrams,
            "height": height,
            "width": width,
            "filename": os.path.abspath(
                os.path.join(out_dir, phase, os.path.basename(filename)[:-3] + 'png'))
        }

        with open(os.path.join(out_dir, phase, os.path.basename(filename)[:-3] + 'json'), 'w') as fp:
            fp.write(simplejson.dumps(data, indent=2))
        return True

    except Exception as exp:
        print("Skipping {} : {}".format(filename, exp))
        return False
