# wave_tool
Этот репозиторий содержит консольное приложение для работы с WAV аудиофайлами, позволяющее изменять скорость и громкость воспроизведения, а также осуществлять расшифровку речи в текст с помощью модели Whisper-tiny. Приложение предназначено для анализа и обработки аудиоданных в офлайн-режиме и поддерживает распознавание речи на разных языках.

Использование
-------------

### Модификация аудио

Для изменения скорости воспроизведения и громкости аудиофайла выполните следующую команду:

`python main.py modify <path_to_audio_file> <speed_multiplier> <volume_change_dB>`

Примеры:

`python main.py modify audio_ru.wav 2 10`
`python main.py modify audio_en.wav 2 10`


### Расшифровка аудио в текст

Для транскрипции аудио в текст используйте следующую команду:

`python main.py transcribe <path_to_audio_file>`

Примеры:

`python main.py transcribe audio_ru.wav`
`python main.py transcribe audio_en.wav`

Форматы файлов
--------------

*   **Модификация аудио**: Сохраняется как `<original_filename>_modified.wav`
*   **Расшифровка аудио**: Результаты сохраняются в JSON файле `<original_filename>_transcription.json`
