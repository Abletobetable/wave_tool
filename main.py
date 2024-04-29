import sys
import json
from pydub import AudioSegment
from pydub.playback import play
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf

def change_audio_properties(audio_path, speed=1.0, volume_change=0.0):
    audio = AudioSegment.from_wav(audio_path)
    # Изменение скорости воспроизведения
    new_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    })
    # Изменение громкости
    new_audio = new_audio + volume_change
    # Сохранение измененного файла
    output_path = "modified_" + audio_path
    new_audio.export(output_path, format="wav")
    return output_path

def transcribe_audio(audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-tiny"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Загрузка аудиофайла
    speech, rate = sf.read(audio_path)

    # Транскрипция
    result = pipe({"raw": speech, "sampling_rate": rate})

    # Формирование JSON и сохранение в файл
    json_output_path = audio_path + "_transcription.json"
    with open(json_output_path, 'w') as json_file:
        json.dump({"transcription": result["text"]}, json_file)

    return json_output_path

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <function> <file_path> [options]")
        return

    function = sys.argv[1]
    file_path = sys.argv[2]

    if function == "modify":
        speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        volume_change = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        modified_file = change_audio_properties(file_path, speed, volume_change)
        print("Modified audio saved to:", modified_file)
    elif function == "transcribe":
        json_file_path = transcribe_audio(file_path)
        print("Transcription saved in:", json_file_path)
    else:
        print("Invalid function. Use 'modify' or 'transcribe'.")

if __name__ == "__main__":
    main()
