from django.db import models
from picklefield.fields import PickledObjectField

class Agent(models.Model):
    instance = PickledObjectField()
    # actions set
    # number of each type of request (request, train, check)
    # maybe past performance
    # time of creation
    # last update time
    # user specified domain
    # owner

# Actions sets
    # functions
    # features

# Python Functions

# Users and User permissions

