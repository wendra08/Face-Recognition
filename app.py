import streamlit as st
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import joblib
from datetime import datetime

# Fungsi untuk memuat model SVM dan data
def load_model():
    model = joblib.load('svm_face_recognition_model.pkl')  # Muat model SVM
    known_face_encodings = np.load('known_face_encodings.npy', allow_pickle=True)  # Memuat encoding wajah
    known_face_names = np.load('known_face_names.npy')  # Memuat nama orang yang dikenal
    return model, known_face_encodings, known_face_names

# Fungsi untuk mendeteksi wajah
def recognize_faces(frame, model, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_names = []
    
    for face_encoding in face_encodings:
        # Menggunakan model SVM untuk prediksi
        name = "Unknown"
        predictions = model.predict([face_encoding])
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        
        if face_distance[best_match_index] < 0.6:  # Jika cocok dengan threshold
            name = known_face_names[best_match_index]
        
        face_names.append(name)
    
    return face_locations, face_names

# Fungsi untuk menulis log absensi
def mark_attendance(name):
    now = datetime.now()
    time_string = now.strftime('%H:%M:%S')
    date_string = now.strftime('%Y-%m-%d')
    
    df = pd.DataFrame({
        'Name': [name],
        'Date': [date_string],
        'Time': [time_string]
    })
    
    if not os.path.exists('attendance.csv'):
        df.to_csv('attendance.csv', index=False)
    else:
        df.to_csv('attendance.csv', mode='a', header=False, index=False)

# Fungsi untuk memeriksa apakah sudah ada absensi hari ini
def is_already_absent(name):
    if not os.path.exists('attendance.csv'):
        return False
    
    try:
        df = pd.read_csv('attendance.csv')
    except pd.errors.EmptyDataError:
        return False
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    return ((df['Name'] == name) & (df['Date'] == today)).any()

# Fungsi untuk melakukan absensi jika belum ada hari ini
def process_absence(name):
    if not is_already_absent(name):
        mark_attendance(name)
        st.success(f'Absensi untuk {name} berhasil.')
        st.session_state.absence_message_shown = False
    else:
        if not st.session_state.get('absence_message_shown', False):
            st.warning('Anda sudah melakukan absensi hari ini.')
            st.session_state.absence_message_shown = True

# Aplikasi Streamlit
st.title('Absence System')

# Initialize session state for message tracking
if 'absence_message_shown' not in st.session_state:
    st.session_state.absence_message_shown = False

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

# Muat model SVM dan data wajah
model, known_face_encodings, known_face_names = load_model()

cap = cv2.VideoCapture(0)

if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Tidak bisa membuka kamera.")
            break

        face_locations, face_names = recognize_faces(frame, model, known_face_encodings, known_face_names)

        # Gambar kotak di sekitar wajah yang terdeteksi dan berikan label nama
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Proses absensi jika wajah dikenal dan belum absen hari ini
            if name != "Unknown":
                process_absence(name)
        
        FRAME_WINDOW.image(frame, channels='BGR')

else:
    cap.release()
    st.write('Kamera tidak aktif.')

# Tombol absensi manual
name = st.text_input('Masukkan Nama untuk Absen')
if st.button('Absen'):
    if name:
        process_absence(name)
    else:
        st.warning('Nama harus diisi untuk melakukan absensi.')
