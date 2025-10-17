import joblib
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import re
from nltk.corpus import cmudict
import textstat
from nltk import pos_tag
import numpy as np


class LinguisticMetrics:
    def __init__(self, text):
        self.text = text

    def average_sen_len(self):
        sentences = sent_tokenize(self.text)
        total_words = 0
        total_sentences = len(sentences)

        for sentence in sentences:
            words = word_tokenize(sentence)
            total_words += len(words)

        return total_words / total_sentences

    def average_word_len(self):
        words = word_tokenize(self.text)
        total_length = 0
        total_words = len(words)

        for word in words:
            total_length += len(word)

        return total_length / total_words

    def word_sentence_diff(self):
        total_words = len(word_tokenize(self.text))
        total_sentences = len(sent_tokenize(self.text))

        return total_words / total_sentences

    def vocabulary_richness(self):
        words = word_tokenize(self.text.lower())
        words = [word for word in words if word.isalpha()]

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
        metrics = [self.average_sen_len(), self.average_word_len(), self.word_sentence_diff(),
                self.vocabulary_richness(), self.syllables_per_word_cmu(),
                self.complex_conjunctions_freq(), self.punctuation_density(),
                textstat.flesch_reading_ease(self.text) ,textstat.dale_chall_readability_score(self.text)]
        return [round(x, 2) for x in metrics]


X_test = [18.54545455, 3.990196078, 18.54545455,
          0.576271186, 1.314917127, 0.265536723,
          0.192090395, 0.062146893, 0, 0.027217742,
          79.25114271, 8.148543474]
rf_loaded = joblib.load("random_forest_model.pkl")
y_pred = rf_loaded.predict(X_test)