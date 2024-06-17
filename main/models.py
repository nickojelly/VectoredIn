from django.db import models

# Create your models here.
class JobPosting(models.Model):
    title = models.CharField(max_length=200)
    company = models.CharField(max_length=200)
    description = models.TextField()
    products = models.TextField()
    vector = models.TextField() 