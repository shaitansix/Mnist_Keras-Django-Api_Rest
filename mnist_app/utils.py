import os
import io
import base64
from PIL import Image
import numpy as np
from keras.models import load_model
from django.conf import settings
from .models import AnnModel, HyperparamsModel, DataParamsModel

def saveAnn(self, ann): 
  hyperparams = HyperparamsModel.objects.get_or_create(
    activation = ann.activation, 
    learning_rate = ann.learning_rate, 
    epochs = ann.epochs
  )

  params = DataParamsModel.objects.get_or_create(
    batch_size = ann.batch_size, 
    ratio_train = ann.ratio_train
  )

  new_ann = AnnModel.objects.create(
    arquitecture = ','.join(ann.arquitecture), 
    loss = ann.test_loss, 
    accuracy = ann.test_acc, 
    data_params = params[0], 
    hyperparams = hyperparams[0]
  )

  model_root = os.path.join('data', 'keras_models', f'ann-{new_ann.id}.keras')
  ann.model.save(model_root)
  new_ann.model_file = model_root
  new_ann.save()

  return new_ann

def loadAnn(self, id): 
  ann_model = AnnModel.objects.get(id = id)
  model_root = os.path.join(settings.BASE_DIR, ann_model.model_file)
  ann = load_model(model_root)
  return ann

def getImageMatrix(image_base64): 
  image = image_base64.split('base64,')[1]

  img_bytes = base64.b64decode(image)
  img = Image.open(io.BytesIO(img_bytes))
  img_resized = img.resize((28, 28), Image.LANCZOS)
  img_resized = img_resized.convert('L') 

  img_matrix = np.array(img_resized)
  img_matrix = 255 - img_matrix
  img_matrix = img_matrix.astype(np.float32) / 255
  img_matrix = img_matrix.reshape(784, )
  return img_matrix