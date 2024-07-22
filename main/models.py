from django.db import models
import numpy as np
# Create your models here.
class JobPosting(models.Model):
    title = models.CharField(max_length=200)
    company = models.CharField(max_length=200)
    description = models.TextField()
    products = models.TextField()
    vector = models.TextField() 
    


class QueryEmbedding(models.Model):
    query = models.CharField(max_length=200)
    rag_query = models.TextField(null=True)
    query_embedding = models.TextField()
    rag_query_embedding = models.TextField(null=True)

class Summaries(models.Model):
    querys = models.TextField(null=True)
    summaries = models.TextField(null=True)

class JobListing(models.Model):
    job_id = models.UUIDField(null=True)
    wv_uuid = models.TextField(null=True)
    company_name = models.TextField(null=True)
    company_id = models.UUIDField(null=True)
    title = models.TextField(null=True)
    text = models.TextField(null=True)
    annotations = models.TextField(null=True)
    vector = models.TextField(null=True)
    entity_indices = models.TextField(null=True)

class SubComponemtEmbeddings(models.Model):
    index = models.IntegerField(null=True)
    embedding = models.BinaryField(null=True)
    text = models.TextField(null=True)
    entity = models.TextField(null=True)

    def set_embedding(self, array):
        self.embedding = array.tobytes()

    def get_embedding(self):
        return np.frombuffer(self.embedding, dtype=np.float32) 