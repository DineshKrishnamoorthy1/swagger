# Create your views here.
import uuid
from datetime import datetime
from pathlib import Path

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from attribution_train.AttributionTrainSchema import AttributionTrainSchema
from attribution_train.Serializer import OnlineAttributionTrainSerializer
from kvp_attribution.core import attribution_train, fileGenerator, prepare_data
# Create your views here.
from kvp_attribution.invoicenet import FIELD_TYPES
from rest_framework.generics import GenericAPIView


class OnlineAttributionTrain(GenericAPIView):
    serializer_class = OnlineAttributionTrainSerializer
    def post(self,request,format=None):
        unq_dir = datetime.now().strftime("%b-%d-%Y-%H-%M-%S") + '-' + str(uuid.uuid1()) + '/'
        input_file = request.data.get('inputFile')
        field = request.data.get('field')
        value = request.data.get('value')
        root_dir = request.data.get('rootDir')
        train_data_path = Path(root_dir + unq_dir + 'train_data/')
        batch_size = request.data.get('batchSize')
        processed_data_path = Path(root_dir + unq_dir + 'process_data/')
        restore = request.data.get('restore')
        steps = request.data.get('steps')
        early_stop_steps = request.data.get('earlyStopSteps')
        model_data_path = Path(root_dir + unq_dir + 'model/')

        train_data_path.mkdir(parents=True, exist_ok=True)
        processed_data_path.mkdir(parents=True, exist_ok=True)
        model_data_path.mkdir(parents=True, exist_ok=True)

        train_data_abs_path = str(train_data_path.absolute()) + '/'
        processed_data_abs_path = str(processed_data_path.absolute()) + '/'
        model_data_abs_path = str(model_data_path.absolute()) + '/'

        provided_fields = dict()
        provided_fields[field] = FIELD_TYPES["general"]

        fileGenerator(input_file, field, value, train_data_abs_path)
        prepare_data(train_data_abs_path, processed_data_abs_path, provided_fields)

        attribution_train_output = attribution_train(model_data_abs_path, processed_data_abs_path, field, batch_size,
                                                     restore, steps,
                                                     early_stop_steps, provided_fields)

        response = {'id': unq_dir,
                    'status': status.HTTP_200_OK, 'AttributionTrain': True,
                    'attributionTrainOutput': attribution_train_output}

        return Response(response)

