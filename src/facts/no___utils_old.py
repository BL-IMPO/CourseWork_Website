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
    DALE = f.readlines()

# Test to ensure it runs
MODEL.predict(np.zeros((1, 16)))

class LinguisticMetrics:
    def __init__(self, text):
        self.text = text
        self.words = word_tokenize(self.text.lower())
        self.sentences = sent_tokenize(self.text)

    def total_words(self):
        #words = word_tokenize(self.text)
        total_words = len(self.words)

        return total_words

    def total_sen(self):
        total_sentences = len(self.sentences)

        return total_sentences

    def average_sen_len(self):
        total_words = 0
        total_sentences = len(self.sentences)

        for sentence in self.sentences:
            words = word_tokenize(sentence)
            total_words += len(words)

        return total_words / total_sentences

    def average_word_len(self):
        total_length = 0
        total_words = len(self.words)

        for word in self.words:
            total_length += len(word)

        return total_length / total_words

    def word_sentence_diff(self):
        total_words = len(self.words)
        total_sentences = len(self.sentences)

        return total_words / total_sentences

    def vocabulary_richness(self):
        words = [word for word in self.words if word.isalpha()]

        if len(words) == 0:
            return 0

        total_words = len(words)
        unique_words = len(set(words))

        return unique_words / total_words

    def count_syllables(self, word):
        """
        Простой счетчик слогов в английских словах
        """
        word = word.lower().strip()
        if len(word) <= 3:
            return 1

        # Подсчитаем группы гласных
        vowels = 'aeiouy'
        count = 0

        # Учитываем исключения
        if word.endswith('e'):
            word = word[:-1]

        if word.endswith('le') and len(word) > 2:
            count += 1

        # Подсчитаем гласные
        prev_char_vowel = False
        for char in word:
            if char in vowels:
                if not prev_char_vowel:
                    count += 1
                prev_char_vowel = True
            else:
                prev_char_vowel = False

        return max(1, count)


    def count_syllables_cmu(self, word):
        """
        Подсчитаем слоги с помощью cmu
        """
        cmu_dict = cmudict.dict()
        word = word.lower()
        if word in cmu_dict:
            # Получим произношения для каждого слова
            pronunciation = cmu_dict[word][0]
            # Подсчитем гласные в произношении
            return len([ph for ph in pronunciation if ph[-1].isdigit()])
        else:
            # Если не удастся, то вернемся к простой функции
            return self.count_syllables(word)

    def syllables_per_word_cmu(self):
        words = re.findall(r'\b[a-zA-Z]+\b', self.text.lower())

        if len(words) == 0:
            return 0

        total_syllables = sum(self.count_syllables_cmu(word) for word in words)
        return total_syllables / len(words)

    def get_pos_ratios(self):
        """
        Подсчитывает количество сущ., прил., глаголов относительно количества слов
        """
        # получаем только слова
        words = [word for word in word_tokenize(self.text) if word.isalpha()]

        if not words:
            return 0.0, 0.0, 0.0

        # Получаем POS теги
        tags = [tag for _, tag in pos_tag(words)]

        # Подсчитываем количество слов для каждого POS
        nouns = len([tag for tag in tags if tag.startswith('NN')])
        verbs = len([tag for tag in tags if tag.startswith('VB')])
        adjs = len([tag for tag in tags if tag.startswith('JJ')])

        total = len(words)
        return nouns / total, verbs / total, adjs / total

    def complex_conjunctions_freq(self):
        """Часточность сложных союзов в тексте"""
        conjunctions = {'although', 'however', 'nevertheless', 'moreover',
                        'furthermore', 'therefore', 'consequently', 'thus',
                        'hence', 'meanwhile', 'otherwise', 'despite', 'unless'}

        words = word_tokenize(self.text.lower())
        words = [w for w in words if w.isalpha()]

        if not words:
            return 0

        conj_count = sum(1 for word in words if word in conjunctions)
        return conj_count / len(words)


    def punctuation_density(self):
        """Знаки пунктуации относительно всего текста"""
        punct_marks = len(re.findall(r'[.,;:!?\-—()"\'/]', self.text))
        return punct_marks / len(self.text) if self.text else 0


    def flesch_reading_ease(self):
        return 206.835 - 1.015 * self.average_sen_len() - 84.6 * self.syllables_per_word_cmu()

    def dale_chall_readability_score(self):

        difficult_words = 0

        for word in self.words:
            if word in DALE:
                difficult_words += 1

        diff_to_normal_words = (difficult_words / len(self.words))  * 100

        dale_score = 0.1579 * diff_to_normal_words + 0.0496 * self.word_sentence_diff()

        if diff_to_normal_words > 5:
            dale_score += 3.6365

        return dale_score

    def smog_index(self):

        if len(self.sentences) == 0:
            return 0

        # Count polysyllabic words (words with 3 or more syllables)
        polysyllabic_count = 0

        for sentence in self.sentences:
            words = word_tokenize(sentence)
            for word in words:
                # Clean the word - keep only alphabetic characters
                clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
                if len(clean_word) > 0:
                    syllable_count = self.count_syllables_cmu(clean_word)
                    if syllable_count >= 3:
                        polysyllabic_count += 1

        # Apply SMOG formula
        sentences_count = min(len(self.sentences), 30)  # SMOG uses exactly 30 sentences
        if polysyllabic_count == 0:
            return 0

        smog_score = 1.043 * (polysyllabic_count * (30 / sentences_count)) ** 0.5 + 3.1291
        return round(smog_score, 2)

    def automated_readability_index(self):
        if len(self.words) == 0 or len(self.sentences) == 0:
            return 0

        # Count total characters (letters only, excluding punctuation and spaces)
        total_characters = 0
        for word in self.words:
            # Count only alphabetic characters in each word
            total_characters += len(re.findall(r'[a-zA-Z]', word))

        # Calculate ratios
        characters_per_word = total_characters / len(self.words)
        words_per_sentence = len(self.words) / len(self.sentences)

        # Apply ARI formula
        ari_score = (4.71 * characters_per_word) + (0.5 * words_per_sentence) - 21.43
        return round(ari_score, 2)

    def coleman_liau_index(self):
        if len(self.words) == 0:
            return 0

        # Count total characters (letters only, excluding punctuation and spaces)
        total_characters = 0
        for word in self.words:
            # Count only alphabetic characters in each word
            total_characters += len(re.findall(r'[a-zA-Z]', word))

        # Calculate metrics per 100 words
        characters_per_100_words = (total_characters / len(self.words)) * 100
        sentences_per_100_words = (len(self.sentences) / len(self.words)) * 100

        # Apply Coleman-Liau formula
        coleman_liau_score = (0.0588 * characters_per_100_words) - (0.296 * sentences_per_100_words) - 15.8
        return round(coleman_liau_score, 2)

    def gunning_fog_index(self):
        if len(self.words) == 0 or len(self.sentences) == 0:
            return 0

        # Count complex words (words with 3 or more syllables)
        complex_word_count = 0
        words = [word for word in self.words if word.isalpha()]

        for word in words:
            # Skip common exceptions: proper nouns (capitalized), familiar jargon, etc.
            # For simplicity, we'll count all words with 3+ syllables as complex
            syllable_count = self.count_syllables_cmu(word)
            if syllable_count >= 3:
                complex_word_count += 1

        # Calculate ratios
        words_per_sentence = len(words) / len(self.sentences)
        complex_word_percentage = (complex_word_count / len(words)) * 100 if len(words) > 0 else 0

        # Apply Gunning Fog formula
        fog_index = 0.4 * (words_per_sentence + complex_word_percentage)
        return round(fog_index, 2)

    def get_all_metrics(self):
        """
        Returns all metrics for text
        """
        pos = self.get_pos_ratios()
        metrics = [self.average_sen_len(), self.average_word_len(), self.word_sentence_diff(),
                self.vocabulary_richness(), self.syllables_per_word_cmu(),
                pos[0], pos[1], pos[2],
                self.complex_conjunctions_freq(), self.punctuation_density(),
                self.flesch_reading_ease() ,self.dale_chall_readability_score(),
                self.smog_index(), self.automated_readability_index(),
                self.coleman_liau_index(), self.gunning_fog_index(),
                ]

        return [round(x, 2) for x in metrics]


class NNPredict:
    def __init__(self, text):
        self.target = 0
        self.text = text

    def __predict(self):
        metrics = LinguisticMetrics(self.text).get_all_metrics()

        x_predict = np.array(metrics)
        x_predict = x_predict.reshape(1, -1)
        x_predict = SCALER_X.fit_transform(x_predict)

        self.target = SCALER_Y.inverse_transform(MODEL.predict(x_predict))
        print(self.target)
    def predict_level(self):
        self.__predict()

        lexile = self.target  # e.g., 750, 950, etc.
        if lexile < 500:
            return "very_easy"
        elif lexile < 800:
            return "easy"
        elif lexile < 1100:
            return "medium"
        else:
            return "hard"


def run_readability_analysis(text_content):
    model = NNPredict(text_content)
    metrics = LinguisticMetrics(text_content)
    result = [model.predict_level(), metrics.total_words(), metrics.total_sen(), ]
    metrics = metrics.get_all_metrics()

    for metric in metrics:
        result.append(metric)
    print(result)
    return result
