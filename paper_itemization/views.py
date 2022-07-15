
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter
from django.http import HttpResponse
from elasticsearch import Elasticsearch
from rest_framework import status
from rest_framework.generics import GenericAPIView

from paper_itemization.Serializers import PaperSerializer


audit_split_started =''
audit_output_dir_creation=''
audit_input_file=''
audit_output_filepath=''
audit_data_time=''
audit_file_extension=''
audit_directory_creation=''
audit_pdf_pageno =''
audit_split_failed=''



def create_dir(dir_path):
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.info("Created output directory {}".format(dir_path))
        global audit_directory_creation
        audit_directory_creation='The Directory is Created Successfully'
        print(audit_directory_creation)


def split_input_file(file_name, file_path, output_dir_path, output_file_paths):
    try:
        global audit_split_started
        audit_split_started='started'
        print(audit_split_started)
        input_file_name = os.path.splitext(file_name)[0]
        pdf_output_dir = output_dir_path + "/" + input_file_name + "/"
        create_dir(pdf_output_dir)
        input_pdf = PdfFileReader(open(file_path, "rb"))
        for page_no in range(input_pdf.numPages):
            output = PdfFileWriter()
            output.addPage(input_pdf.getPage(page_no))
            global audit_pdf_pageno
            audit_pdf_pageno = 'splited the Page'+page_no

            print(audit_pdf_pageno)
            new_file_name = pdf_output_dir + str(input_file_name) + ("-page-no-%s.pdf" % page_no)
            with open(new_file_name, "wb") as outputStream:
                output.write(outputStream)
                outputStream.close()
            output_file_paths.append(new_file_name)
    except Exception as e:
        global audit_split_failed
        audit_split_failed ='Error While Splitting the Input File'
        print(audit_split_failed)
    logging.error("Error while splitting the  input file {}".format(e))

class PaperItemization(GenericAPIView):
    serializer_class = PaperSerializer

    logging.info(serializer_class.data)

    def post(self, request, format=None):
        try:
            method_type = request.method
            logging.info("Starting the PDF splitting process")
            if method_type == 'POST':
                input_file = request.data.get('inputFilePath')
                logging.info(input_file)
                global audit_input_file
                audit_input_file='Input file path Received'
                print(audit_input_file)
                output_path = request.data.get('outputDir')
                logging.info(output_path)
                global audit_output_filepath
                audit_output_filepath='Output File Path Received'
                print(audit_output_filepath)
                current_date_time = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
                global audit_data_time
                audit_data_time ='Created On'+current_date_time
                print(audit_data_time)
                output_dir_path = output_path + "/paper_itemization/" + str(random.randint(1501, 3501)) + str(
                    current_date_time)
                logging.info("Output directory path {}".format(output_dir_path))
                create_dir(output_dir_path)
                global audit_output_dir_creation
                audit_output_dir_creation='Output Directory Created'
                print(audit_output_dir_creation)
                output_file_paths = list()
                if os.path.isdir(input_file):
                    base_path = input_file
                    for file_name in os.listdir(input_file):
                        input_file = base_path + '/' + file_name
                        split_input_file(file_name, input_file, output_dir_path, output_file_paths)
                        output_dir_json = json.dumps(output_file_paths)
                        return HttpResponse(output_dir_json, status=status.HTTP_201_CREATED)
                elif os.path.isfile(input_file):
                    file_name, file_extension = os.path.splitext(input_file)
                    file_name = file_name.split('/')
                    file_name = file_name[-1]
                    if file_extension == '.pdf':
                        global audit_file_extension
                        audit_file_extension='The File is PDF'
                        print(audit_file_extension)
                        split_input_file(file_name, input_file, output_dir_path, output_file_paths)
                    else:
                        output_file_paths.append(input_file)
                    output_dir_json = json.dumps(output_file_paths)
                    return HttpResponse(output_dir_json, status=status.HTTP_201_CREATED)
                        # return Response(request.data)
            else:
                logging.error("The given input is not a file or directory")
        except Exception as e:
            logging.error("Error in preprocessing the input file {}".format(e))
        return HttpResponse("Incorrect method call...", status=status.HTTP_400_BAD_REQUEST)



class Auditable(object):
    def __init__(self, audit_split_started: str, audit_output_dir_creation: str, audit_input_file: str, audit_output_filepath: str, audit_data_time: str, audit_file_extension: str, audit_directory_creation: str, audit_pdf_pageno: str, audit_split_failed: str):
        self.audit_split_failed = audit_split_failed
        self.audit_pdf_pageno = audit_pdf_pageno
        self.audit_directory_creation = audit_directory_creation
        self.audit_file_extension = audit_file_extension
        self.audit_data_time = audit_data_time
        self.audit_output_filepath = audit_output_filepath
        self.audit_input_file = audit_input_file
        self.audit_output_dir_creation = audit_output_dir_creation
        self.audit_split_started = audit_split_started


user = Auditable(audit_split_started=audit_split_started, audit_output_dir_creation=audit_output_dir_creation, audit_input_file=audit_input_file, audit_output_filepath=audit_output_filepath, audit_data_time =audit_data_time, audit_file_extension=audit_file_extension, audit_directory_creation=audit_directory_creation, audit_pdf_pageno=audit_pdf_pageno, audit_split_failed=audit_split_failed)
print(user)
json_data = json.dumps(user.__dict__)
print(json_data)
print(Auditable(**json.loads(json_data)))

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


resp = es.index(index="example", id=1, document=json_data)
print(resp['result'])




