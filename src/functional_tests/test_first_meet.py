import os

from selenium import webdriver
from django.contrib.staticfiles.testing import StaticLiveServerTestCase

class GreetingsTest(StaticLiveServerTestCase):
    def setUp(self):
        self.browser = webdriver.Firefox()

    def tearDown(self):
        self.browser.quit()
        super().tearDown()

    def test_first_meeting(self):
        # Zanyl goes to the website and sees greeting message
        self.browser.get('http://localhost:8000')

        self.assertIn("Hello", self.browser.title)

