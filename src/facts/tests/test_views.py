import pytest
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

# Mark this module to use Django’s database and settings
pytestmark = pytest.mark.django_db


@pytest.fixture
def client(client):
    """Return Django test client."""
    return client


def test_analyze_view_get(client):
    """Test GET request renders the analyze page."""
    url = reverse("analyze")
    response = client.get(url)
    assert response.status_code == 200
    assert "analyze.html" in [t.name for t in response.templates]


def test_analyze_view_post_with_text(monkeypatch, client):
    """Test POST request with plain text input."""

    # Mock readability function
    def mock_analysis(text):
        return {"score": 80, "feedback": "Mocked feedback"}

    # Patch it
    monkeypatch.setattr("your_app.views.run_readability_analysis", mock_analysis)

    url = reverse("analyze")
    data = {"text_input": "This is a simple text for testing."}
    response = client.post(url, data)

    assert response.status_code == 200
    assert b"Mocked feedback" in response.content
    assert b"This is a simple text for testing." in response.content


def test_analyze_view_post_with_file(monkeypatch, client, tmp_path):
    """Test POST request with uploaded file."""

    # Mock readability function
    def mock_analysis(text):
        return {"score": 90, "feedback": "File processed successfully"}

    monkeypatch.setattr("your_app.views.run_readability_analysis", mock_analysis)

    url = reverse("analyze")

    # Create fake text file
    file_content = b"Some file content for readability testing."
    test_file = SimpleUploadedFile("test.txt", file_content, content_type="text/plain")

    response = client.post(url, {"file_input": test_file})

    assert response.status_code == 200
    assert b"File processed successfully" in response.content
