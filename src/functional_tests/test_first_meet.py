import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class GreetingsTest(StaticLiveServerTestCase):
    def setUp(self):
        self.browser = webdriver.Firefox()
        self.wait = WebDriverWait(self.browser, 10)

    def tearDown(self):
        self.browser.quit()
        super().tearDown()

    def test_first_meeting(self):
        # Zanyl goes to the website and sees greeting message
        self.browser.get('http://localhost:8000')

        analyze_link = self.browser.find_element(By.LINK_TEXT, "Analyze a Book")
        self.assertIsNotNone(analyze_link)

        # She clicks it abd sees another page
        ## Find and click the "Analyze a Book" button/link
        analyze_link = self.wait.until(
            EC.element_to_be_clickable((By.LINK_TEXT, "Analyze a Book"))
        )
        analyze_link.click()

        ## Wait for and verify navigation to analyze page
        self.wait.until(EC.url_contains('analyze'))

        ## Verify we're on the analysis page by checking page content
        ## Option 1: Check URL pattern
        self.assertIn('analyze', self.browser.current_url)

        # She sees buttons Paste Text and Upload File
        paste_tab = self.wait.until(
            EC.presence_of_element_located((By.ID, "paste-tab"))
        )
        upload_tab = self.wait.until(
            EC.presence_of_element_located((By.ID, "upload-tab"))
        )

        self.assertIsNotNone(paste_tab)
        self.assertIsNotNone(upload_tab)

        # Verify the tabs have correct text
        self.assertEqual(paste_tab.text, "Paste Text")
        self.assertEqual(upload_tab.text, "Upload File")

        # Check that Paste Text tab is active by default
        self.assertIn("active", paste_tab.get_attribute("class"))

        # She clicks Upload File tab
        upload_tab.click()

        # Verify Upload File tab becomes active
        self.wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "#upload-tab.active"))
        )
        self.assertIn("active", upload_tab.get_attribute("class"))

        # Verify Paste Text tab is no longer active
        self.assertNotIn("active", paste_tab.get_attribute("class"))

        # She clicks Paste Text tab again
        paste_tab.click()

        # Verify Paste Text tab is active again
        self.wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "#paste-tab.active"))
        )
        self.assertIn("active", paste_tab.get_attribute("class"))

