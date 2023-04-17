from django.db import models
from picklefield.fields import PickledObjectField
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _

from apprentice.planners.fo_planner import Operator as Opp


class Agent(models.Model):
    """
    Agents are the meat of the Apprentice Learner API that instantiate the
    various learning mechanisms.
    """
    instance = PickledObjectField()
    uid = models.CharField(max_length=50, primary_key=True)
    name = models.CharField(max_length=200, blank=True)
    num_act = models.IntegerField(default=0)
    num_train = models.IntegerField(default=0)
    num_check = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def inc_act(self):
        self.num_act = self.num_act + 1

    def inc_train(self):
        self.num_train = self.num_train + 1

    def inc_check(self):
        self.num_check = self.num_check + 1

    def __str__(self):
        return str(self.instance)
    #     skills = {}

    #     try:
    #         skill_dict = self.instance.skills
    #         for label in skill_dict:
    #             for i, how in enumerate(skill_dict[label]):
    #                 name = label
    #                 if i > 0:
    #                     name = "%s-%i" % (label, i+1)
    #                 skills[name] = {}
    #                 skills[name]['where'] = skill_dict[label][how]['where_classifier']
    #                 skills[name]['when'] = skill_dict[label][how]['when_classifier']
    #                 skills[name]['how'] = how

    #     except:
    #         pass

    #     return "Agent {0} - {1} : {2}".format(self.pk, self.name, len(skills))

    # def generate_trees(self):
    #     import pydotplus
    #     from sklearn import tree
    #     from sklearn.externals.six import StringIO

    #     for label in self.instance.skills:
    #         for n, how in enumerate(self.instance.skills[label]):
    #             pipeline = self.instance.skills[label][how]['when_classifier']

    #             dv = pipeline.steps[0][1]
    #             dt = pipeline.steps[1][1]

    #             dot_data = StringIO()
    #             tree.export_graphviz(dt, out_file=dot_data,
    #                                  feature_names=dv.feature_names_,
    #                                  class_names=["Don't Fire Rule",
    #                                               "Fire Rule"],
    #                                  filled=True, rounded=True,
    #                                  special_characters=True)
    #             graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #             graph.write_png("decisiontrees/%s-%i.png" % (label, n))

    class Meta:
        ordering = ('-updated',)

    # user specified domain
    # owner

# Users and User permissions
