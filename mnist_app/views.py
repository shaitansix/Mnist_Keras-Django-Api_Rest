import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .dl_model import AnnKeras
from .utils import saveAnn, loadAnn, getImageMatrix

@api_view(['POST'])
def create_ann(request): 
  data = request.data
  ann = AnnKeras(**data)
  ann.train()

  new_ann = saveAnn(ann)
  res = { 
    'state': 'success',
    'result': {
      'id': new_ann.id, 
      'loss': ann.test_loss, 
      'accuracy': ann.test_acc, 
      'history_loss': ann.getHistoryLoss(), 
      'history_acc': ann.getHistoryAcc()
    }, 
    'message': 'ANN created and trained successfully'
  }

  return Response(res)

@api_view(['POST'])
def classify_image(request, *args, **kwargs): 
  id = kwargs['id']
  image_base64 = request.data.get('image', '')

  ann = loadAnn(id)
  img_matrix = getImageMatrix(image_base64)

  x = np.array([img_matrix])
  pred = ann.predict(x, verbose = 0)
  num = np.argmax(pred)

  res = {
    'state': 'success',
    'result': {
      'id': id, 
      'number': num, 
      'probability': pred[0][num]
    }, 
    'message': 'Image classified successfully'
  }

  return Response(res)