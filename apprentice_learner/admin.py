from django.contrib import admin
from apprentice_learner.models import Agent
from apprentice_learner.models import ActionSet
from apprentice_learner.models import PyFunction

class AgentAdmin(admin.ModelAdmin):
    pass
class ActionSetAdmin(admin.ModelAdmin):
    pass
class PyFunctionAdmin(admin.ModelAdmin):
    pass

# Register your models here.
admin.site.register(Agent, AgentAdmin)
admin.site.register(ActionSet, AgentAdmin)
admin.site.register(PyFunction, AgentAdmin)
