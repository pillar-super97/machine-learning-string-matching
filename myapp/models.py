from django.db import models
import uuid


class BaseModel(models.Model):
    """A base model to deal with all the abstract level model creations"""

    # uuid field
    uuid = models.UUIDField(primary_key=True,
                            default=uuid.uuid4,
                            editable=False)

    # date fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class MatchedRecords(models.Model): 
    source = models.CharField(max_length=200, null=True, blank=True)
    target = models.CharField(max_length=200, null=True, blank=True)
    confidence = models.IntegerField(default=0)


class StringRecords(BaseModel):
    source_format_name = models.CharField(max_length=200, null=True, blank=True)
    target_format_name = models.CharField(max_length=200, null=True, blank=True)
    overall_confidence = models.IntegerField(default=0)
    match_results = models.ManyToManyField(MatchedRecords, blank=True)


class TrainingService(models.Model):
    source = models.CharField(max_length=200, null=True, blank=True)
    target = models.CharField(max_length=200, null=True, blank=True)