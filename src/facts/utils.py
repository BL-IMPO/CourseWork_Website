from django.db.models.fields import return_None
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import re
from nltk.corpus import cmudict
import textstat
from nltk import pos_tag
import numpy as np
from tensorflow.keras.models import load_model
import os
from django.conf import settings
from keras.models import load_model
import joblib


# Pre-load CMU dict and stopwords once at module level
CMU_DICT = cmudict.dict()
STOP_WORDS = set(stopwords.words('english'))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "text_complexity_model.keras")
SCALER_X_PATH = os.path.join(MODEL_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "scaler_y.pkl")
DALE_PATH = os.path.join(CURRENT_DIR, "dale.txt")

MODEL = load_model(MODEL_PATH)
SCALER_X = joblib.load(SCALER_X_PATH)
SCALER_Y = joblib.load(SCALER_Y_PATH)

with open(DALE_PATH, "r") as f:
    DALE = set(line.strip().lower() for line in f)  # Use set for O(1) lookups

# Test to ensure it runs
MODEL.predict(np.zeros((1, 16)), verbose=0)  # Disable verbose


class LinguisticMetrics:
    def __init__(self, text):
        self.text = text
        # Precompute tokens once and reuse
        self.words = word_tokenize(text.lower())
        self.sentences = sent_tokenize(text)
        self.alpha_words = [word for word in self.words if word.isalpha()]
        self.total_alpha_words = len(self.alpha_words)
        self.total_words = len(self.words)
        self.total_sentences = len(self.sentences)

        # Precompute syllable counts for all words
        self._syllable_cache = {}
        self._precompute_syllables()

    def _precompute_syllables(self):
        """Precompute syllable counts for all unique words"""
        unique_words = set(self.alpha_words)
        for word in unique_words:
            self._syllable_cache[word] = self._count_syllables_fast(word)

    def _count_syllables_fast(self, word):
        """Optimized syllable counter"""
        word = word.lower().strip()
        if len(word) <= 3:
            return 1

        # Use CMU dict if available, otherwise fallback to regex
        if word in CMU_DICT:
            pronunciation = CMU_DICT[word][0]
            return len([ph for ph in pronunciation if ph[-1].isdigit()])

        # Fast regex-based fallback
        word = re.sub(r'[^a-z]', '', word)
        if word.endswith('e'):
            word = word[:-1]

        vowels = re.findall(r'[aeiouy]+', word)
        return max(1, len(vowels)) if vowels else 1

    def total_words(self):
        return self.total_words

    def total_sen(self):
        return self.total_sentences

    def average_sen_len(self):
        if self.total_sentences == 0:
            return 0
        return self.total_words / self.total_sentences

    def average_word_len(self):
        if not self.words:
            return 0
        total_length = sum(len(word) for word in self.words)
        return total_length / len(self.words)

    def word_sentence_diff(self):
        if self.total_sentences == 0:
            return 0
        return self.total_words / self.total_sentences

    def vocabulary_richness(self):
        if not self.alpha_words:
            return 0
        return len(set(self.alpha_words)) / len(self.alpha_words)

    def syllables_per_word_cmu(self):
        if not self.alpha_words:
            return 0

        # Use precomputed syllable counts
        total_syllables = sum(self._syllable_cache.get(word, 1) for word in self.alpha_words)
        return total_syllables / len(self.alpha_words)

    def get_pos_ratios(self):
        if not self.alpha_words:
            return 0.0, 0.0, 0.0

        # Batch POS tagging
        tags = [tag for _, tag in pos_tag(self.alpha_words)]

        nouns = verbs = adjs = 0
        for tag in tags:
            if tag.startswith('NN'):
                nouns += 1
            elif tag.startswith('VB'):
                verbs += 1
            elif tag.startswith('JJ'):
                adjs += 1

        total = len(self.alpha_words)
        return nouns / total, verbs / total, adjs / total

    def complex_conjunctions_freq(self):
        if not self.alpha_words:
            return 0

        conjunctions = {'although', 'however', 'nevertheless', 'moreover',
                        'furthermore', 'therefore', 'consequently', 'thus',
                        'hence', 'meanwhile', 'otherwise', 'despite', 'unless'}

        # Use set intersection for faster counting
        conj_count = len(conjunctions.intersection(self.alpha_words))
        return conj_count / len(self.alpha_words)

    def punctuation_density(self):
        if not self.text:
            return 0
        punct_marks = len(re.findall(r'[.,;:!?\-—()"\'/]', self.text))
        return punct_marks / len(self.text)

    def flesch_reading_ease(self):
        asl = self.average_sen_len()
        spw = self.syllables_per_word_cmu()
        return 206.835 - 1.015 * asl - 84.6 * spw

    def dale_chall_readability_score(self):
        if not self.words:
            return 0

        # Use set for O(1) lookups
        difficult_words = sum(1 for word in self.words if word in DALE)
        diff_to_normal_words = (difficult_words / len(self.words)) * 100

        dale_score = 0.1579 * diff_to_normal_words + 0.0496 * self.word_sentence_diff()

        if diff_to_normal_words > 5:
            dale_score += 3.6365

        return dale_score

    def smog_index(self):
        if self.total_sentences == 0:
            return 0

        # Use precomputed syllable counts
        polysyllabic_count = 0
        for word in self.alpha_words:
            if self._syllable_cache.get(word, 1) >= 3:
                polysyllabic_count += 1

        sentences_count = min(self.total_sentences, 30)
        if polysyllabic_count == 0:
            return 0

        smog_score = 1.043 * (polysyllabic_count * (30 / sentences_count)) ** 0.5 + 3.1291
        return round(smog_score, 2)

    def automated_readability_index(self):
        if not self.words or not self.sentences:
            return 0

        # Count characters using pre-tokenized words
        total_characters = sum(len(re.findall(r'[a-zA-Z]', word)) for word in self.words)

        characters_per_word = total_characters / len(self.words)
        words_per_sentence = len(self.words) / len(self.sentences)

        ari_score = (4.71 * characters_per_word) + (0.5 * words_per_sentence) - 21.43
        return round(ari_score, 2)

    def coleman_liau_index(self):
        if not self.words:
            return 0

        total_characters = sum(len(re.findall(r'[a-zA-Z]', word)) for word in self.words)
        characters_per_100_words = (total_characters / len(self.words)) * 100
        sentences_per_100_words = (self.total_sentences / len(self.words)) * 100

        coleman_liau_score = (0.0588 * characters_per_100_words) - (0.296 * sentences_per_100_words) - 15.8
        return round(coleman_liau_score, 2)

    def gunning_fog_index(self):
        if not self.alpha_words or not self.sentences:
            return 0

        # Use precomputed syllable counts
        complex_word_count = 0
        for word in self.alpha_words:
            if self._syllable_cache.get(word, 1) >= 3:
                complex_word_count += 1

        words_per_sentence = len(self.alpha_words) / self.total_sentences
        complex_word_percentage = (complex_word_count / len(self.alpha_words)) * 100

        fog_index = 0.4 * (words_per_sentence + complex_word_percentage)
        return round(fog_index, 2)

    def get_all_metrics(self):
        """Returns all metrics for text - optimized to avoid redundant calculations"""
        pos = self.get_pos_ratios()

        # Precompute reusable values
        asl = self.average_sen_len()
        awl = self.average_word_len()
        wsd = self.word_sentence_diff()
        vr = self.vocabulary_richness()
        spw = self.syllables_per_word_cmu()
        ccf = self.complex_conjunctions_freq()
        pd = self.punctuation_density()
        fre = 206.835 - 1.015 * asl - 84.6 * spw  # Direct calculation
        dcrs = self.dale_chall_readability_score()
        si = self.smog_index()
        ari = self.automated_readability_index()
        cli = self.coleman_liau_index()
        gfi = self.gunning_fog_index()

        metrics = [asl, awl, wsd, vr, spw, pos[0], pos[1], pos[2], ccf, pd,
                   fre, dcrs, si, ari, cli, gfi]

        return [round(x, 2) for x in metrics]


class NNPredict:
    def __init__(self, text):
        self.target = 0
        self.text = text

    def __predict(self):
        metrics = LinguisticMetrics(self.text).get_all_metrics()

        x_predict = np.array(metrics).reshape(1, -1)
        x_predict = SCALER_X.transform(x_predict)  # Use transform, not fit_transform

        self.target = SCALER_Y.inverse_transform(MODEL.predict(x_predict, verbose=0))

    def predict_level(self):
        self.__predict()

        lexile = self.target[0][0]  # Extract scalar value
        if lexile < 500:
            return "very_easy"
        elif lexile < 800:
            return "easy"
        elif lexile < 1100:
            return "medium"
        else:
            return "hard"


def run_readability_analysis(text_content):
    metrics_calculator = LinguisticMetrics(text_content)
    model = NNPredict(text_content)

    result = [model.predict_level(), metrics_calculator.total_words, metrics_calculator.total_sentences]
    metrics = metrics_calculator.get_all_metrics()

    result.extend(metrics)
    return result
