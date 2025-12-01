#  Text Readability Prediction Model

[cite_start]This repository contains the source code and research for the coursework **"Building a model for predicting text readability based on linguistic characteristics"**[cite: 7].

[cite_start]The goal of this project is to create a Machine Learning model capable of predicting the difficulty level of English texts (approximating the **Lexile Band** [cite: 46][cite_start]) using open-source linguistic features, without relying on paid proprietary tools[cite: 43, 45].

##  Key Features

[cite_start]The project implements feature extraction for **20+ linguistic characteristics** [cite: 275-341], including:
* [cite_start]**Surface Features:** Average Sentence Length, Average Word Length (in characters and syllables)[cite: 179, 183].
* [cite_start]**Phonetic Analysis:** Syllable counting using the CMU Pronouncing Dictionary[cite: 295].
* [cite_start]**Morphological Analysis:** Part-of-Speech ratios (Nouns, Verbs, Adjectives) using NLTK[cite: 300].
* [cite_start]**Syntactic Features:** Frequency of complex conjunctions and punctuation density[cite: 306, 307].
* [cite_start]**Traditional Metrics:** Implementation of Flesch Reading Ease, SMOG, Gunning Fog, Dale-Chall, and others[cite: 313].

##  Dataset

[cite_start]The project utilizes the **CommonLit Ease of Readability (CLEAR) Corpus**, which contains approximately 5,000 text excerpts annotated with readability scores and Lexile bands[cite: 155].

##  Tech Stack

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **NLP:** NLTK, RE (Regular Expressions)
* [cite_start]**Machine Learning:** Scikit-learn, XGBoost, TensorFlow (Keras)[cite: 343].

##  Results

A comparative analysis of four machine learning models was conducted. [cite_start]The **XGBRegressor** model demonstrated the best performance[cite: 494].

| Model | MAE | RMSE | $R^2$ Score |
|-------|-----|------|----------|
| Linear Regression | 0.34 | 0.20 | 0.798 |
| Random Forest | 0.31 | 0.17 | 0.826 |
| **XGBoost (Best)** | **0.31** | **0.17** | **0.828** |
| MLP (Neural Net) | 0.31 | 0.41 | 0.825 |

[cite_start]*Results based on the test set[cite: 464, 465].*

## ⚙️ Setup & Installation

Follow these steps to set up the project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BL-IMPO/CourseWork_Website.git](https://github.com/BL-IMPO/CourseWork_Website.git)
    cd CourseWork_Website
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn nltk xgboost tensorflow
    ```

4.  **Download NLTK Data:**
    [cite_start]The feature extraction scripts require specific NLTK corpora[cite: 295, 297, 301]. Run the following python code:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('cmudict')
    ```

5.  **Required Files:**
    [cite_start]Ensure the file `dale.txt` (containing the list of 3000 common words for the Dale-Chall formula) is located in the root directory[cite: 320].

6.  **Run the Notebook:**
    ```bash
    jupyter notebook
    ```
