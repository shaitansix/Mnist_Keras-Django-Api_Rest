import os
import io
import base64
import tempfile
from PIL import Image
import numpy as np
from keras.models import load_model
from .models import AnnModel, HyperparamsModel, DataParamsModel

def kerasToBytes(ann): 
  with tempfile.NamedTemporaryFile(suffix = '.keras', delete = False) as tmp_file:
    tmp_path = tmp_file.name
  ann.model.save(tmp_path)

  with open(tmp_path, 'rb') as file:
    model_bytes = file.read()
  os.unlink(tmp_path)

  return model_bytes

def BytesToKeras(ann_model): 
  with tempfile.NamedTemporaryFile(suffix = '.keras', delete = False) as tmp_file:
    tmp_path = tmp_file.name
    tmp_file.write(ann_model.model_file)

  return tmp_path

def saveAnn(ann): 
  hyperparams = HyperparamsModel.objects.get_or_create(
    activation = ann.activation, 
    learning_rate = ann.learning_rate, 
    epochs = ann.epochs
  )

  params = DataParamsModel.objects.get_or_create(
    batch_size = ann.batch_size, 
    ratio_train = ann.ratio_train
  )
  
  model_bytes = kerasToBytes(ann)

  new_ann = AnnModel.objects.create(
    model_file = model_bytes, 
    arquitecture = ','.join(ann.arquitecture), 
    loss = ann.test_loss, 
    accuracy = ann.test_acc, 
    data_params = params[0], 
    hyperparams = hyperparams[0]
  )

  return new_ann

def loadAnn(id): 
  ann_model = AnnModel.objects.get(id = id)

  tmp_path = BytesToKeras(ann_model)
  ann = load_model(tmp_path)
  os.unlink(tmp_path)
  
  return ann_model, ann

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