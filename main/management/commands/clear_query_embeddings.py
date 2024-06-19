from django.core.management.base import BaseCommand
from main.models import QueryEmbedding

class Command(BaseCommand):
    help = 'Clears all data from the QueryEmbedding model'

    def handle(self, *args, **options):
        QueryEmbedding.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully cleared data from QueryEmbedding model'))