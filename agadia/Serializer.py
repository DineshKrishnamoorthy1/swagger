from rest_framework import serializers

class AgadiaSerializer(serializers.Serializer):
    inputFile = serializers.CharField()
    modelPath = serializers.CharField()
    key = serializers.CharField()
    outputPat = serializers.CharField()

