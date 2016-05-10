from django.db import models
from picklefield.fields import PickledObjectField

class Agent(models.Model):
    instance = PickledObjectField()
