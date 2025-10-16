// scripts.js

// Tab functionality
document.addEventListener('DOMContentLoaded', function() {
    // Tab switching
    const tabTriggers = document.querySelectorAll('.tab-trigger');
    const tabContents = document.querySelectorAll('.tab-content');
    
    if (tabTriggers.length > 0) {
        tabTriggers.forEach(trigger => {
            trigger.addEventListener('click', function() {
                const targetId = this.getAttribute('data-target');
                
                // Update active tab
                tabTriggers.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                
                // Show target content
                tabContents.forEach(content => {
                    content.classList.add('hidden');
                    content.classList.remove('active');
                });
                
                const targetContent = document.getElementById(targetId);
                if (targetContent) {
                    targetContent.classList.remove('hidden');
                    targetContent.classList.add('active');
                }
            });
        });
    }
    
    // Word count for textarea
    const textInput = document.getElementById('text-input');
    const wordCount = document.getElementById('word-count');
    
    if (textInput && wordCount) {
        textInput.addEventListener('input', function() {
            const text = this.value.trim();
            const words = text ? text.split(/\s+/).length : 0;
            wordCount.textContent = `${words} words`;
        });
    }
    
    // File input handling
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    
    if (fileInput && fileName) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                fileName.textContent = `Selected: ${this.files[0].name}`;
            } else {
                fileName.textContent = '';
            }
        });
    }
    
    // Analyze button functionality
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsSection = document.getElementById('results-section');
    const analyzeAnotherBtn = document.getElementById('analyze-another-btn');
    
    if (analyzeBtn && resultsSection) {
        analyzeBtn.addEventListener('click', function() {
            // Show loading state
            this.textContent = 'Analyzing...';
            this.disabled = true;
            
            // Simulate analysis delay
            setTimeout(() => {
                // Show results
                resultsSection.classList.remove('hidden');
                
                // Scroll to results
                resultsSection.scrollIntoView({ behavior: 'smooth' });
                
                // Reset button
                this.textContent = 'Analyze Text';
                this.disabled = false;
            }, 1500);
        });
    }
    
    if (analyzeAnotherBtn && resultsSection) {
        analyzeAnotherBtn.addEventListener('click', function() {
            // Hide results
            resultsSection.classList.add('hidden');
            
            // Reset form
            if (textInput) textInput.value = '';
            if (wordCount) wordCount.textContent = '0 words';
            if (fileInput) fileInput.value = '';
            if (fileName) fileName.textContent = '';
            
            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});