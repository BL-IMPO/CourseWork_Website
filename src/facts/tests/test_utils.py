import pytest

import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from facts.utils import LinguisticMetrics, NNPredict


def test_class_metrics_returns_all_metrics():
    text = """When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.
            The floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches.
            At each end of the room, on the wall, hung a beautiful bear-skin rug.
            These rugs were for prizes, one for the girls and one for the boys. And this was the game.
            The girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole.
            This would have been an easy matter, but each traveller was obliged to wear snowshoes."""
    metrics = LinguisticMetrics(text)
    expected = [18.55, 3.99, 18.55, 0.58, 1.31, 0.26, 0.19, 0.06, 0.00, 0.02, 79.25, 8.14]
    assert metrics.get_all_metrics() == pytest.approx(expected, abs=0.03)

def test_if_random_forest_can_gues_difficulty_level():
    text = """When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.
                The floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches.
                At each end of the room, on the wall, hung a beautiful bear-skin rug.
                These rugs were for prizes, one for the girls and one for the boys. And this was the game.
                The girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole.
                This would have been an easy matter, but each traveller was obliged to wear snowshoes."""

    model = NNPredict(text)

    assert model.predict_level() in ["very_easy", "easy", "medium", "hard"]