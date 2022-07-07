
from abc import ABC

import field as field
from rest_marshmallow import Schema

class HandWrittenAttributionSchema(Schema, ABC):
    inputfilepath = field.String()
    modelfilepath = field.string()