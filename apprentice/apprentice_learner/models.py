from django.db import models
from picklefield.fields import PickledObjectField
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _

from planners.fo_planner import Operator as Opp

# class PyFunction(models.Model):
#     name = models.CharField(max_length=200)
#     fun_def = models.TextField()
#
#     def __str__(self):
#         return self.name
#
# class ActionSet(models.Model):
#     name = models.CharField(max_length=200, unique=True)
#     features = models.ManyToManyField(PyFunction, blank=True,
#                                       related_name="feature_action_sets")
#     functions = models.ManyToManyField(PyFunction, blank=True,
#                                       related_name="function_action_sets")
#
#     def get_feature_dict(self):
#         features = {}
#         for feature in self.features.all():
#             temp = {}
#             exec(feature.fun_def, temp)
#             features[feature.name] = temp[feature.name]
#         return features
#
#     def get_function_dict(self):
#         functions = {}
#         for function in self.functions.all():
#             temp = {}
#             exec(function.fun_def, temp)
#             functions[function.name] = temp[function.name]
#         return functions
#
#     def __str__(self):
#         return self.name

def valid_py(value):
    """
    TODO - make this more sophisticated for validating the actual stucture of an Operator
    """
    try:
        eval(value)
    except:
        raise ValidationError(
            _('%(value)s is not a valid python statment'),
            params={'value':value},)

class Operator(models.Model):
    """
    A model to contain all of the pieces of an fo_planner.Operator.
    Note that what the fo_planner.Operator calls "name" is "head" here.
    TODO - change fo_planner.Operator so the above isn't confusing
    """
    name = models.CharField(max_length=200, blank=True)
    head = models.TextField(validators=[valid_py], blank=True)
    conditions = models.TextField(validators=[valid_py], blank=True)
    effects = models.TextField(validators=[valid_py], blank=True)

    def compile(self):
        """
        Takes the textual Operator components and compiles them into an fo_planner.Operator
        """
        return Opp(eval(self.head), eval(self.conditions), eval(self.effects))

    def __str__(self):
        return self.name

class Project(models.Model):
    """
    I hate the name Project but I need something to serve as an Agent
    container that can be used to instantiate common settings.
    """
    name = models.CharField(max_length=200, blank=False, default='DEFAULT_PROJECT')
    numerical_epsilon = models.FloatField(default=0.0)
    feature_set = models.ManyToManyField(Operator, blank=True, related_name='feature_sets')
    function_set = models.ManyToManyField(Operator, blank=True, related_name='function_sets')

    def compile_features(self):
        return [op.compile() for op in self.feature_set.all()]

    def compile_functions(self):
        return [op.compile() for op in self.function_set.all()]

    def __str__(self):
        return self.name

class Agent(models.Model):
    """
    Agents are the meat of the Apprentice Learner API that instantiate the
    various learning mechanisms.
    """
    instance = PickledObjectField()
    # action_set = models.CharField(max_length=200, blank=False)
    # action_set = models.ForeignKey(ActionSet, on_delete=models.CASCADE)
    name = models.CharField(max_length=200, blank=True)
    num_request = models.IntegerField(default=0)
    num_train = models.IntegerField(default=0)
    num_check = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True)


    def inc_request(self):
        self.num_request = self.num_request + 1

    def inc_train(self):
        self.num_train = self.num_train + 1

    def inc_check(self):
        self.num_check = self.num_check + 1

    def __str__(self):
        skills = {}

        try:
            skill_dict = self.instance.skills
            for label in skill_dict:
                for i, how in enumerate(skill_dict[label]):
                    name = label
                    if i > 0:
                        name = "%s-%i" % (label, i+1)
                    skills[name] = {}
                    skills[name]['where'] = skill_dict[label][how]['where_classifier']
                    skills[name]['when'] = skill_dict[label][how]['when_classifier']
                    skills[name]['how'] = how

        except:
            pass

        return "Agent {0} - {1} : {2}".format(self.pk, self.name, len(skills))

    def generate_trees(self):
        import pydotplus
        from sklearn import tree
        from sklearn.externals.six import StringIO

        for label in self.instance.skills:
            for n, how in enumerate(self.instance.skills[label]):
                pipeline = self.instance.skills[label][how]['when_classifier']

                dv = pipeline.steps[0][1]
                dt = pipeline.steps[1][1]

                dot_data = StringIO()
                tree.export_graphviz(dt, out_file=dot_data,
                                     feature_names=dv.feature_names_,
                                     class_names=["Don't Fire Rule",
                                                  "Fire Rule"],
                                     filled=True, rounded=True,
                                     special_characters=True)
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                graph.write_png("decisiontrees/%s-%i.png" % (label, n))

    class Meta:
        ordering = ('-updated',)

    # user specified domain
    # owner

# Users and User permissions
