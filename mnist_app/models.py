from django.db import models

class HyperparamsModel(models.Model): 
  activation = models.CharField(max_length = 10)
  learning_rate = models.FloatField()
  epochs = models.IntegerField()
  created = models.DateTimeField(auto_now_add = True)

  class Meta: 
    verbose_name = 'hyperparam'
    verbose_name_plural = 'hyperparams'

  def __str__(self): 
    return f'Hyperparams - {self.id}'

class DataParamsModel(models.Model): 
  batch_size = models.IntegerField()
  ratio_train = models.FloatField()
  created = models.DateTimeField(auto_now_add = True)

  class Meta: 
    verbose_name = 'data_param'
    verbose_name_plural = 'data_params'

  def __str__(self): 
    return f'Data params - {self.id}'

class AnnModel(models.Model): 
  model_file = models.CharField(max_length = 100, null = True, blank = True)
  arquitecture = models.CharField(max_length = 20)
  loss = models.FloatField()
  accuracy = models.FloatField()
  data_params = models.ForeignKey(DataParamsModel, on_delete = models.CASCADE)
  hyperparams = models.ForeignKey(HyperparamsModel, on_delete = models.CASCADE)
  created = models.DateTimeField(auto_now_add = True)

  class Meta: 
    verbose_name = 'ann'
    verbose_name_plural = 'anns'
  
  def __str__(self): 
    return f'ANN - {self.id}'