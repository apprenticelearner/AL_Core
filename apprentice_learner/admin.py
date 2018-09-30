from django import forms
from django.contrib import admin
from apprentice_learner.models import Agent
from apprentice_learner.models import Project
from apprentice_learner.models import Operator

from codemirror2.widgets import CodeMirrorEditor


class AgentAdmin(admin.ModelAdmin):
    pass

# class OperatorAdminForm(forms.ModelForm):
#     model = Operator
#     class Meta:
#         fields = '__all__'
#         widgets = {
#             'code': PythonEditor(attrs={'style':'width: 90%; height: 100%;'}),
#         }

class OperatorAdmin(admin.ModelAdmin):
    def formfield_for_dbfield(self, db_field, **kwargs):
        if db_field.attname in ['head', 'conditions', 'effects']:
            kwargs['widget'] = CodeMirrorEditor(options={'mode':'python', 'lineNumbers':True})
        return super(OperatorAdmin, self).formfield_for_dbfield(db_field, **kwargs)



# Register your models here.
admin.site.register(Agent, AgentAdmin)
admin.site.register(Project)
admin.site.register(Operator, OperatorAdmin)
