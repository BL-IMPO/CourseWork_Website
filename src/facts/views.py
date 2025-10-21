from django.shortcuts import render
from django.views.generic import TemplateView
from django.views import View
from django.core.files.storage import default_storage

from facts.utils import run_readability_analysis


class HomeView(TemplateView):
    template_name = "facts/index.html"


class AnalyzeView(View):
    template_name = "facts/analyze.html"

    def get(self, request):
        """Render the empty analyze page."""
        return render(request, self.template_name)

    def post(self, request):
        """Handle text input or file upload."""
        text_input = request.POST.get("text_input", "")
        file_obj = request.FILES.get("file_input")

        text_content = text_input

        if file_obj:
            # Save file temporarily
            file_path = default_storage.save(file_obj.name, file_obj)

            # Read file content (safely)
            with default_storage.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()

            # Delete file after reading
            default_storage.delete(file_path)

        # Run analysis
        results = run_readability_analysis(text_content)

        return render(request, self.template_name, {
            "results": results,
            "text": text_content
        })