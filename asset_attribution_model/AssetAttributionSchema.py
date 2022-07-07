from abc import ABC

from rest_marshmallow import Schema
import field as field

class AssetAttributionSchema(Schema, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inputFile = field.string()
        modelPath = field.string()
        modelType = field.string()
        key = field.string()


