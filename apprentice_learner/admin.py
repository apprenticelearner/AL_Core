from django.contrib import admin
from apprentice_learner.models import Agent

class AgentAdmin(admin.ModelAdmin):
    pass

# Register your models here.
admin.site.register(Agent, AgentAdmin)
