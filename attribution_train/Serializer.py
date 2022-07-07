from rest_framework import serializers


class OnlineAttributionTrainSerializer(serializers.Serializer):
    inputfilepath = serializers.CharField()
    field_name = serializers.CharField()
    value = serializers.IntegerField    ()
    root_dir = serializers.CharField()
    batch_size = serializers.IntegerField()
    restore = serializers.CharField()
    steps = serializers.IntegerField()