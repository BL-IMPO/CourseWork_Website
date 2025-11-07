import pytest

from facts.models import Data


@pytest.mark.django_db
def test_model_gets_text_metrics_and_complexity_level():
    # Create sample text for testing
    sample_text = """
    This is a sample text for testing readability analysis. It contains multiple sentences 
    to ensure proper analysis of various metrics. The text should be long enough to trigger 
    all the analysis functions including average sentence length, word length, and vocabulary 
    richness. This will help verify that all metrics are calculated correctly.

    Another paragraph with different structure and word choices. Testing complex conjunctions 
    and punctuation density is important for comprehensive analysis. The model should handle 
    various text characteristics and provide accurate complexity levels.
    """

    # Create Data instance with text
    data = Data.objects.create(
        text=sample_text,
        # Add other required fields if your model has them
    )

    # Test that the model has the expected attributes/methods
    assert hasattr(data, 'average_sen_len')
    assert hasattr(data, 'average_word_len')
    assert hasattr(data, 'word_sentence_diff')
    assert hasattr(data, 'vocabulary_richness')
    assert hasattr(data, 'syllables_per_word_cmu')
    assert hasattr(data, 'complex_conjunctions_freq')
    assert hasattr(data, 'punctuation_density')


