import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter
from django.http import HttpResponse
from requests import Response
from rest_framework import status
from rest_framework.decorators import api_view

from paper_itemization.PaperItemSchema import PaperSchema


import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from urllib import request

from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter
from django.http import HttpResponse
from rest_framework import status
from rest_framework .generics import GenericAPIView
from rest_framework.decorators import api_view

from paper_itemization.Serializers import PaperSerializer


def create_dir(dir_path):
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.info("Created output directory {}".format(dir_path))


def split_input_file(file_name, file_path, output_dir_path, output_file_paths):
    try:
        input_file_name = os.path.splitext(file_name)[0]
        pdf_output_dir = output_dir_path + "/" + input_file_name + "/"
        create_dir(pdf_output_dir)
        input_pdf = PdfFileReader(open(file_path, "rb"))
        for page_no in range(input_pdf.numPages):
            output = PdfFileWriter()
            output.addPage(input_pdf.getPage(page_no))
            new_file_name = pdf_output_dir + str(input_file_name) + ("-page-no-%s.pdf" % page_no)
            with open(new_file_name, "wb") as outputStream:
                output.write(outputStream)
                outputStream.close()
            output_file_paths.append(new_file_name)
    except Exception as e:
        logging.error("Error while splitting the  input file {}".format(e))



class PaperItemization(GenericAPIView):
    serializer_class = PaperSerializer
    def post(self, request , format=None):
        try:
            method_type = request.method
            logging.info("Starting the PDF splitting process")
            if method_type == 'POST':

                input_file = request.data.get('inputFilePath')
                output_path = request.data.get('outputDir')
                current_date_time = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
                output_dir_path = output_path + "/paper_itemization/" + str(random.randint(1501, 3501)) + str(current_date_time)
                logging.info("Output directory path {}".format(output_dir_path))
                create_dir(output_dir_path)
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
                            split_input_file(file_name, input_file, output_dir_path, output_file_paths)
                    else:
                        output_file_paths.append(input_file)
                        output_dir_json = json.dumps(output_file_paths)
                        return HttpResponse(output_dir_json, status=status.HTTP_201_CREATED)
                        return Response(request.data)
            else:
                logging.error("The given input is not a file or directory")
        except Exception as e:
                        logging.error("Error in preprocessing the input file {}".format(e))
        return HttpResponse("Incorrect method call...", status=status.HTTP_400_BAD_REQUEST)


