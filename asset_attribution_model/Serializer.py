from rest_framework import serializers

class doAssetAttributionSerializer(serializers.Serializer):
    inputFile = serializers.CharField()
    modelPath = serializers.CharField()
    modelType = serializers.CharField()
    key = serializers.CharField()
    outputDir = serializers.CharField()

