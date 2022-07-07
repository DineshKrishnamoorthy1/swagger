from abc import ABC

import field as field
from rest_marshmallow import Schema

class AttributionTrainSchema(Schema):
    def __init__(self):
        super().__init__()
        inputfilepath = field.string()
        field_name = field.string()
        value = field.Integer()
        root_dir = field.string()
        batch_size = field.Integer()
        restore = field.string()
        steps = field.Integer()
