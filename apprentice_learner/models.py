from django.db import models
from picklefield.fields import PickledObjectField

class PyFunction(models.Model):
    name = models.CharField(max_length=200)
    fun_def = models.TextField()

    def __str__(self):
        return self.name

class ActionSet(models.Model):
    name = models.CharField(max_length=200, unique=True)
    features = models.ManyToManyField(PyFunction, blank=True,
                                      related_name="feature_action_sets")
    functions = models.ManyToManyField(PyFunction, blank=True,
                                      related_name="function_action_sets")

    def get_feature_dict(self):
        features = {}
        for feature in self.features.all():
            temp = {}
            exec(feature.fun_def, temp)
            features[feature.name] = temp[feature.name]
        return features

    def get_function_dict(self):
        functions = {}
        for function in self.functions.all():
            temp = {}
            exec(function.fun_def, temp)
            functions[function.name] = temp[function.name]
        return functions

    def __str__(self):
        return self.name

class Agent(models.Model):
    instance = PickledObjectField()
    action_set = models.ForeignKey(ActionSet, on_delete=models.CASCADE)
    name = models.CharField(max_length=200, blank=True)
    num_request = models.IntegerField(default=0) 
    num_train = models.IntegerField(default=0)
    num_check = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def inc_request(self):
        self.num_request = self.num_request + 1

    def inc_train(self):
        self.num_train = self.num_train + 1

    def inc_check(self):
        self.num_check = self.num_check + 1

    def __str__(self):
        skills = {} 

        try:
            sd = self.instance.skills
            for label in sd:
                for i, how in enumerate(sd[label]):
                    name = label
                    if i > 0:
                        name = "%s-%i" % (label, i+1)
                    skills[name] = {}
                    skills[name]['where'] = sd[label][how]['where_classifier']
                    skills[name]['when'] = sd[label][how]['when_classifier']
                    skills[name]['how'] = how

        except:
            pass

        return "Agent %i - %s : %s" % (self.id, self.name, skills)

    class Meta:
        ordering = ('-updated',)
   
    # user specified domain
    # owner

# Users and User permissions

