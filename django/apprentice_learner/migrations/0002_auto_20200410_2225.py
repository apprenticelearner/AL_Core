# Generated by Django 2.1.11 on 2020-04-11 02:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('apprentice_learner', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='project',
            name='feature_set',
        ),
        migrations.RemoveField(
            model_name='project',
            name='function_set',
        ),
        migrations.RemoveField(
            model_name='agent',
            name='project',
        ),
        migrations.DeleteModel(
            name='Project',
        ),
    ]
