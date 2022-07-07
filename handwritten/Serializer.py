from rest_framework import serializers

class HandWrittenSerializer(serializers.Serializer):
    inputfilepath = serializers.CharField()
    modelfilepath = serializers.CharField()
