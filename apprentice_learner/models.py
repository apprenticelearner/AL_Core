from django.db import models
from picklefield.fields import PickledObjectField

class PyFunction(models.Model):
    name = models.CharField(max_length=200)
    fun_def = models.TextField()

    def __str__(self):
        return self.name

class ActionSet(models.Model):
    name = models.CharField(max_length=200)
    features = models.ManyToManyField(PyFunction, blank=True, related_name="feature_action_sets")
    function = models.ManyToManyField(PyFunction, blank=True, related_name="function_action_sets")
    def __str__(self):
        return self.name

class Agent(models.Model):
    instance = PickledObjectField()
    action_set = models.ForeignKey(ActionSet, on_delete=models.CASCADE)
    name = models.CharField("agent's name", max_length=200)
    num_request = models.IntegerField(default=0) 
    num_train = models.IntegerField(default=0)
    num_check = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
   
    # user specified domain
    # owner

# Users and User permissions

