# Generated by Django 4.1.13 on 2024-06-20 16:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Summaries',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('querys', models.TextField(null=True)),
                ('summaries', models.TextField(null=True)),
            ],
        ),
    ]
