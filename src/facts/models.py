from django.db import models

# Create your models here.

class Data(models.Model):
    text = models.TextField(default="")
    predicted_level = models.CharField(default="", max_length=12)
    total_words = models.IntegerField(default=0)
    total_sen = models.IntegerField(default=0)
    average_sen_len = models.FloatField(default=0.0)
    average_word_len = models.FloatField(default=0.0)
    word_sentence_diff = models.FloatField(default=0.0)
    vocabulary_richness = models.FloatField(default=0.0)
    syllables_per_word_cmu = models.FloatField(default=0.0)
    noun_ratio = models.FloatField(default=0.0)
    verb_ratio = models.FloatField(default=0.0)
    adjective_ratio = models.FloatField(default=0.0)
    complex_conjunctions_freq = models.FloatField(default=0.0)
    punctuation_density = models.FloatField(default=0.0)
    flesch_reading_ease = models.FloatField(default=0.0)
    dale_chall_readability_score = models.FloatField(default=0.0)
    smog_index = models.FloatField(default=0.0)
    automated_readability_index = models.FloatField(default=0.0)
    coleman_liau_index = models.FloatField(default=0.0)
    gunning_fog = models.FloatField(default=0.0)


