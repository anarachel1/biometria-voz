# pip install sounddevice wave
import os
import wave
import keras
import wavio
import pickle
import librosa
import threading
import numpy as np
import sounddevice as sd
from tkinter import PhotoImage, Tk, Canvas, Button, Label, messagebox
from PIL import Image, ImageTk  


def record_audio(duracao=3, taxa_de_amostragem=22050, canais=2):
    print("Gravação iniciada")
    recording = sd.rec(int(duracao * taxa_de_amostragem), samplerate=taxa_de_amostragem, channels=canais, dtype='int16')
    sd.wait()  # Aguardar até que a gravação esteja completa
    print("Gravação finalizada")

    # Salvar a gravação em um arquivo WAV
    wavio.write("recording.wav", recording, taxa_de_amostragem, sampwidth=2)

        
# Cria nova thread para evitar que a GIU trave durante gravacao!
def thread_recording():
    t = threading.Thread(target=record_audio)
    t.start()
    

# Função que simula o processo de autenticação com biometria de voz
def authenticate_with_voice():
    print("loading audio...")
    audio = "recording.wav"           
    audio, sr = librosa.load(audio, sr=22000, duration=3)

    # gera o spectograma
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    log_S = librosa.amplitude_to_db(S, ref=np.max) #log scale

    x = np.array(log_S)
    x = x.reshape(1,*x.shape, 1)
    
    # load the model
    print("loading model...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    print("making the prediction ...")
    result = np.argmax(model.predict(x))
    if result == 0:
        messagebox.showinfo('Autenticação', 'Acesso liberado 🎊')
    else:
        messagebox.showerror('Autenticação', ' 🚨 Acesso negado 🚨')



# Interface de usuário Tkinter abaixo ---------------

root = Tk()
root.title('Login com Biometria de Voz')
root.geometry('400x450')  # Ajuste para o tamanho apropriado da sua imagem

# Carregamento da imagem de fundo usando Pillow
try:
    bg_image = Image.open('img/img-bg.jpg')
    bg_image = bg_image.resize((400, 460))  # Resize the image to fit the window size
    bg_photo = ImageTk.PhotoImage(bg_image)
except IOError as e:
    print(e)
    root.destroy()
    raise

# Criação de um canvas para a imagem de fundo e ajustar para cobrir a janela inteira
canvas = Canvas(root, width=400, height=450)
canvas.pack(fill='both', expand=True)
canvas.create_image(200, 220, image=bg_photo, anchor='center')

# Adicionar labels e botões ao canvas
instructions = canvas.create_text(200, 90, text=" Material sensível ",  fill='#73C2FB', font=('Gill Sans', 22, 'bold'))
instructions2 = canvas.create_text(200, 120, text="Acesso permitido apenas a Ana Rachel", fill='white', font=('Gill Sans Light', 10))

# Botão de gravação
record_button = Button(root, text='Gravar voz', command=thread_recording)
record_button_window = canvas.create_window(140, 160, anchor='n', window=record_button)

# Botão de análise
analyze_button = Button(root, text='Verificar biometria', command=authenticate_with_voice)
analyze_button_window = canvas.create_window(260, 160, anchor='n', window=analyze_button)

root.mainloop()