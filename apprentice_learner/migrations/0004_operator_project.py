# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2017-09-18 19:36
from __future__ import unicode_literals

from django.db import migrations, models
import python_field.fields


class Migration(migrations.Migration):

    dependencies = [
        ('apprentice_learner', '0003_auto_20170509_1408'),
    ]

    operations = [
        migrations.CreateModel(
            name='Operator',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', python_field.fields.PythonCodeField()),
                ('conditions', python_field.fields.PythonCodeField()),
                ('effects', python_field.fields.PythonCodeField()),
            ],
        ),
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('numerical_epsilon', models.FloatField(default=0.0)),
                ('feature_and_function_set', models.CharField(choices=[('tutor knowledge', 'tutor knowledge'), ('stoichiometry', 'stoichiometry'), ('rumbleblocks', 'rumbleblocks'), ('article selection', 'article selection')], default='tutor knowledge', max_length=20)),
            ],
        ),
    ]