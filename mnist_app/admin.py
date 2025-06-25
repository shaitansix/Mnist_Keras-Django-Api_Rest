from django.contrib import admin
from .models import HyperparamsModel, DataParamsModel, AnnModel

class HyperparamsAdmin(admin.ModelAdmin): 
  readonly_fields = ['created']

class DataParamsAdmin(admin.ModelAdmin): 
  readonly_fields = ['created']

class AnnAdmin(admin.ModelAdmin): 
  readonly_fields = ['model_file', 'created']

admin.site.register(HyperparamsModel, HyperparamsAdmin)
admin.site.register(DataParamsModel, DataParamsAdmin)
admin.site.register(AnnModel, AnnAdmin)