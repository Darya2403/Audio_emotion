import tkinter as tk
import tkinter.filedialog
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import tensorflow as tf
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv('dataset.csv', index_col=0)
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('target', axis=1),
                                                   dataset['target'], test_size=0.2, random_state=42)
# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Преобразование целевых меток в целочисленные значения
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
# Создание модели
model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
# Компиляция модели
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Предсказание классов для тестовой выборки
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Вывод метрик обучения
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# Параметры записи аудио
fs = 44100  # частота дискретизации
duration = 5 # длительность записи в секундах

# Функция для записи аудио
def record_audio():
    # Запись аудио
    print("Запись аудио...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # ждем, пока запись закончится

    # Сохранение аудио в файл WAV
    wav.write('audio.wav', fs, audio)

    # Извлечение признаков из аудио с помощью библиотеки librosa
    y, sr = librosa.load('audio.wav', sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30).T, axis=0)
    mfccs = scaler.transform([mfccs])
    print("MFCC Признаки:", mfccs)  # Отобразить MFCC признаки

    # Предсказание эмоции с помощью обученной модели
    mfccs_reshaped = np.reshape(mfccs, (1, 30))
    y_pred = model.predict(mfccs_reshaped).argmax(axis=1)
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprised']
    emotion_predicted = emotions[y_pred[0]]

    # Вывод результата
    print("Предсказанная эмоция:", emotion_predicted)
    label_result.config(text="Предсказанная эмоция: " + emotion_predicted)

# Создание окна приложения
window = tk.Tk()
window.title("Эмоциональный анализ аудио")

# Создание кнопки для записи аудио
button_record = tk.Button(window, text="Записать аудио", command=record_audio)
button_record.pack(pady=10)

# Создание метки для вывода результата
label_result = tk.Label(window, text="")
label_result.pack(pady=10)

# Запуск окна приложения
window.mainloop()