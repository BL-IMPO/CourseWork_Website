from django.shortcuts import render
from django.views.generic import TemplateView
from django.views import View
from django.core.files.storage import default_storage

from facts.utils import run_readability_analysis
from facts.models import Data


class HomeView(TemplateView):
    template_name = "facts/index.html"


class AnalyzeView(View):
    template_name = "facts/analyze.html"

    def __write_to_db(self, text_content, results):
        data = Data(text=text_content, predicted_level=results[0], total_words=results[1],
                    total_sen=results[2], average_sen_len=results[3], average_word_len=results[4],
                    word_sentence_diff=results[5], vocabulary_richness=results[6], syllables_per_word_cmu=results[7],
                    noun_ratio=results[8], verb_ratio=results[9], adjective_ratio=results[10],
                    complex_conjunctions_freq=results[11], punctuation_density=results[12], flesch_reading_ease=results[13],
                    dale_chall_readability_score=results[14], smog_index=results[15], automated_readability_index=results[16],
                    coleman_liau_index=results[17], gunning_fog=results[18])

        data.save()


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
            with default_storage.open(file_path, "rb") as f:
                text_content = f.read().decode("utf-8", errors="ignore")

            # Delete file after reading
            default_storage.delete(file_path)

        # Run analysis
        results = run_readability_analysis(text_content)

        # Save to database
        self.__write_to_db(text_content, results)

        return render(request, self.template_name, {
            "results": results,
            "text": text_content
        })
