from abc import ABC
import field as field
from rest_marshmallow import Schema


class PaperSchema(Schema):
    def __init__(self):
        super().__init__()
        inputFilePath = field.string()
        outputDir = field.string()
