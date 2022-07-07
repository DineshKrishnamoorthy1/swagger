from abc import ABC

from rest_framework import serializers

class PaperSerializer(serializers.Serializer):
    inputFilePath = serializers.CharField()
    outputDir = serializers.CharField()
