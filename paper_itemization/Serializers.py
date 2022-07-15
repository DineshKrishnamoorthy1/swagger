from abc import ABC

from django_elasticsearch_dsl_drf.serializers import DocumentSerializer
from rest_framework import serializers




class PaperSerializer(serializers.Serializer):
    inputFilePath = serializers.CharField()
    outputDir = serializers.CharField()







