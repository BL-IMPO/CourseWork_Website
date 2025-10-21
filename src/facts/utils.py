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


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model", "text_complexity_model.keras")
MODEL = load_model(MODEL_PATH)
MODEL.predict(np.zeros((1, 12)))


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



    def get_all_metrics(self):
        """
        Returns all metrics for text
        """
        pos = self.get_pos_ratios()
        metrics = [self.average_sen_len(), self.average_word_len(), self.word_sentence_diff(),
                self.vocabulary_richness(), self.syllables_per_word_cmu(),
                pos[0], pos[1], pos[2],
                self.complex_conjunctions_freq(), self.punctuation_density(),
                textstat.flesch_reading_ease(self.text) ,textstat.dale_chall_readability_score(self.text),
                ]

        #textstat.smog_index(self.text), textstat.automated_readability_index(self.text),
        #        textstat.coleman_liau_index(self.text), textstat.gunning_fog(self.text),

        return [round(x, 2) for x in metrics]


class NNPredict:
    def __init__(self, text):
        self.target = 0
        self.text = text

    def __predict(self):
        metrics = LinguisticMetrics(self.text).get_all_metrics()
        x_test = np.array(metrics)
        x_test = x_test.reshape(1, -1)

        self.target = MODEL.predict(x_test)

    def predict_level(self):
        self.__predict()

        if self.target > 0:
            return "very_easy"
        elif self.target > -0.5:
            return "easy"
        elif self.target > -1.5:
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

    return result
