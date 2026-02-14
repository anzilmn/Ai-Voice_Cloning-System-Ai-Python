import os
import uuid
import io
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from TTS.api import TTS
from pydub import AudioSegment, effects

# Force Coqui to stay quiet and work
os.environ["COQUI_TOS_AGREED"] = "1"

print("🚀 LOADING FINAL TEACHER-GRADE ENGINE...")
# XTTS v2 is the best, we are just going to tune the parameters for 100% match
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

def index(request):
    if request.method == 'POST' and request.FILES.get('audio_data'):
        try:
            text = request.POST.get('text', '')
            audio_file = request.FILES['audio_data']
            
            samples_dir = os.path.join(settings.MEDIA_ROOT, 'samples')
            outputs_dir = os.path.join(settings.MEDIA_ROOT, 'outputs')
            os.makedirs(samples_dir, exist_ok=True)
            os.makedirs(outputs_dir, exist_ok=True)

            unique_id = uuid.uuid4().hex
            
            # --- THE "VOCAL DNA" CLEANER ---
            raw_data = audio_file.read()
            audio = AudioSegment.from_file(io.BytesIO(raw_data))
            
            # 1. Force Mono and 22050Hz (What XTTS actually uses)
            audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
            
            # 2. Normalize (This makes the AI hear the tiny details of your voice)
            audio = effects.normalize(audio)
            
            # 3. Save as a pure WAV
            sample_name = f"match_{unique_id}.wav"
            speaker_wav = os.path.join(samples_dir, sample_name)
            audio.export(speaker_wav, format="wav")

            # --- THE FINAL CLONE GENERATION ---
            output_name = f"submission_{unique_id}.wav"
            output_path = os.path.join(outputs_dir, output_name)

            print(f"🔥 FORCE-CLONING VOCAL DNA...")
            
            # These specific numbers (0.75, 0.85) usually fix the "Generic AI" sound
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language="en",
                file_path=output_path,
                temperature=0.75,       # Lower = less "random AI"
                length_penalty=1.0, 
                repetition_penalty=5.0, # High = stops robot sounds
                top_k=50,
                top_p=0.85,
                speed=1.0,
                enable_text_splitting=True
            )

            return JsonResponse({'audio_url': f"{settings.MEDIA_URL}outputs/{output_name}"})
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'cloner/index.html')