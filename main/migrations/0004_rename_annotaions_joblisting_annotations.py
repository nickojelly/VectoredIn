# Generated by Django 4.1.13 on 2024-07-19 15:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_joblisting'),
    ]

    operations = [
        migrations.RenameField(
            model_name='joblisting',
            old_name='annotaions',
            new_name='annotations',
        ),
    ]