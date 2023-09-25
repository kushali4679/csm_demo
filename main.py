from fastapi import FastAPI, Request, File, UploadFile,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import speech_recognition as sr
from moviepy.editor import VideoFileClip

from pydub import AudioSegment
from spectralcluster import SpectralClusterer
from resemblyzer import preprocess_wav, VoiceEncoder

import requests
import json
import time
from collections import defaultdict


from pathlib import Path
import os

import assemblyai as aai
import nltk
import openai
from googletrans import Translator
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from google.cloud import secretmanager
from google.cloud import bigquery,storage
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ishaansharma/topic-detector")
model = AutoModelForSequenceClassification.from_pretrained("ishaansharma/topic-detector")
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


key_Path = "cloudkarya-internship-76c41ffa6790.json"
project_id = "cloudkarya-internship"
# bigquery_Client = bigquery.Client.from_service_account_json(key_Path)
# storage_Client = storage.Client.from_service_account_json(key_Path)
# bucket_Name = "pkcsm-raw"



@app.get('/')
def index(request : Request):
    context={"request" : request,
             "predicted_topic": "No Video Uploaded"}
    return templates.TemplateResponse("index1.html",context)


@app.post("/upload_video", response_class=HTMLResponse)
async def upload_video(request : Request, video_file: UploadFile = File(...),text_name: str = Form(...),text_rating: str = Form(...),text_sale: str = Form(...)):
    video_path = f"videos/{video_file.filename}"
    folder_Name = "videos"

    # bucket = storage_Client.get_bucket(bucket_Name)
    # bucket = storage_Client.get_bucket(bucket_Name)
    # blob = bucket.blob(f'{video_file.filename}')
    # blob.upload_from_filename(video_path)


    with open(video_path, "wb") as f:
        f.write(await video_file.read())

    def extract_audio(video_path, output_path):
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_path)
    


    audio_path = "audio.wav"
    extract_audio(video_path, audio_path)

    # Perform topic detection on the conversation
    recognizer = sr.Recognizer()




    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    # Transcribe the audio to text
    conversation = recognizer.recognize_google(audio)

    # Tokenize the conversation
    encoded_input = tokenizer(conversation, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # Forward pass through the model
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    # Get the predicted topic
    predicted_topic = torch.argmax(logits, dim=1).item()

    # Map the predicted topic to the corresponding label
    labels = [
        "Arts_&_culture",
        "Business_&_entrepreneurs",
        "Celebrity_&_pop_culture",
        "Diaries_&_Daily_life",
        "Family",
        "Fashion_&_Style",
        "Film_tv_&_Video",
        "Fitness_&_Health",
        "Food_&_Dining",
        "Gaming",
        "Learning_&_Educational",
        "Music",
        "News_&_Social_concern",
        "Other_hobbies",
        "Relationships",
        "Science_&_Technology",
        "Sports",
        "Travel_&_Adventure",
        "youth_&_student_life"
    ]

    predicted_topic_label = labels[predicted_topic]

   

   #############################################################################
    passage=[]
    YOUR_API_TOKEN = "9ecd2b21976a42fc8032a68cd31f90d3"

    # URL of the file to transcribe
    FILE_URL = "https://drive.google.com/uc?export=download&id=1HCmRWRxaFbVti-P5nuUo4988HxU-d-5S"

    # AssemblyAI transcript endpoint (where we submit the file)
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

    # request parameters where Speaker Diarization has been enabled
    data = {
    "audio_url": FILE_URL,
    "speaker_labels": True,
    "speakers_expected": 3
    }

    # HTTP request headers
    headers={
    "Authorization": "9ecd2b21976a42fc8032a68cd31f90d3",
    "Content-Type": "application/json"
    }

    # submit for transcription via HTTP request
    response = requests.post(transcript_endpoint,
                            json=data,
                            headers=headers)

    # polling for transcription completion
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{response.json()['id']}"
    current_speaker = None
    conversation = defaultdict(list)
    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()

        if transcription_result['status'] == 'completed':
            current_speaker = None
            current_sentence = ""

            for word in transcription_result.get('words', []):
                text = word.get('text')
                speaker = word.get('speaker')

                if text and speaker:
                    if speaker != current_speaker:
                        if current_sentence:
                            passage.append(f"Speaker {current_speaker}: {current_sentence}")
                            print(f"Speaker {current_speaker}: {current_sentence}")
                        current_speaker = speaker
                        current_sentence = text
                    else:
                        current_sentence += " " + text

            # Print the last sentence
            if current_sentence:
                print(f"Speaker {current_speaker}: {current_sentence}")
                passage.append(f"Speaker {current_speaker}: {current_sentence}")
            break
        elif transcription_result['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
        else:
            time.sleep(1)










    #############################################################################


    audio_file = "audio.wav"

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            print("Recognized Text:")
            print(text)
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results from the speech recognition service; {0}".format(e))


    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    x=sia.polarity_scores(text)
    maximum=max(x['neg'],x['neu'],x['pos'])
    if maximum==x['pos']:
        emotion='positive'
    elif maximum==x['neu']:
        emotion='neutral'
    else:
        emotion='negative'
    text=text+'.Here the emotion of the customer and the sales person is '+emotion
    text=text+'.Give us the final summary of the emotion shown by the customer to the sales person and vice versa'
    openai.api_key = ''
    
    
    def chat_with_gpt3(prompt):
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=50,
            temperature=0.6
        )
        return response.choices[0].text.strip()
    print("Welcome to the ChatGPT! Type 'exit' to end the conversation.")
    user_input = text
    response = chat_with_gpt3(user_input)
  
    # query = f"""
    # INSERT INTO `{project_id}.CSM.csm_data`
    # VALUES ('{predicted_topic_label}', '{passage}', '{response}')
    # """
    # job = bigquery_Client.query(query)
    # job.result() 
    
    




    #  # Define the emotions labels
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Function to load the emotion model
    def load_emotion_model():
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        return tf.keras.models.load_model('emotion_model.h5')

    # Function to process frames and detect emotion
    def detect_emotion(frame, emotion_model):
        # Preprocess the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Initialize emotion count dictionary
        emotion_count = {emotion: 0 for emotion in emotions}

        # Draw rectangles around the faces and detect emotion
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            resized_roi = cv2.resize(face_roi, (48, 48))
            normalized_roi = resized_roi / 255.0
            reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
            emotion_prediction = emotion_model.predict(reshaped_roi)
            emotion_label = emotions[np.argmax(emotion_prediction)]

            # Increment emotion count
            emotion_count[emotion_label] += 1

            # Draw rectangle and emotion label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame, emotion_count
    

    def index():
        # Get the uploaded video file

        # Load the emotion model
        emotion_model = load_emotion_model()

        # Read the video file
        video = cv2.VideoCapture(video_path)

        # Initialize emotion count dictionary
        total_emotion_count = {emotion: 0 for emotion in emotions}

        while True:
            # Read a frame from the video
            ret, frame = video.read()
            if not ret:
                break

            # Process the frame and detect emotion
            processed_frame, emotion_count = detect_emotion(frame, emotion_model)

            # Increment total emotion count
            for emotion, count in emotion_count.items():
                total_emotion_count[emotion] += count

            # Display the frame with emotions
            # cv2.imshow("Emotion Detection", processed_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        video.release()
        cv2.destroyAllWindows()

        # Display the emotion count
        for emotion, count in total_emotion_count.items():
            print(f"{emotion}: {count}")

        # Get the emotion with the greatest count
        max_emotion = max(total_emotion_count, key=total_emotion_count.get)
        max_count = total_emotion_count[max_emotion]

        # Define the salesman rating barometer labels
        barometer_labels = ['Worse', 'Bad', 'OK', 'Good', 'Great']

        # Define the corresponding emotion labels for the barometer
        barometer_emotions = ['Angry', 'Disgust', 'Fear', 'Neutral', 'Happy']

        # Determine the rating level based on the emotion
        rating_level = barometer_labels[barometer_emotions.index(max_emotion)]
      
        # Filter emotions with more than 30% of the total count
        filtered_emotions = {emotion: count for emotion, count in total_emotion_count.items() if count > 0.3 * sum(total_emotion_count.values())}

        # Create the pie chart
        labels = filtered_emotions.keys()
        counts = filtered_emotions.values()

        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title('Emotion Distribution')
       
        

            
        
        
        
        # Create a bar chart for the rating level
        rating_levels = barometer_labels
        rating_counts = [0] * len(rating_levels)
        rating_counts[rating_levels.index(rating_level)] = 1
       

        
        
                        # query = """
                        # INSERT INTO `{}.CSM.csm_data`
                        # VALUES (@predicted_topic_label, @passage, @response,@text_name,@text_rating,@text_sale)
                        # """.format(project_id)

                        # job_config = bigquery.QueryJobConfig()
                        # job_config.query_parameters = [
                        #     bigquery.ScalarQueryParameter("predicted_topic_label", "STRING", predicted_topic_label),
                        #     bigquery.ScalarQueryParameter("passage", "STRING", passage),
                        #     bigquery.ScalarQueryParameter("response", "STRING", response),
                        #     bigquery.ScalarQueryParameter("text_name", "STRING", text_name),
                        #     bigquery.ScalarQueryParameter("text_rating", "STRING", rating_levels.index(rating_level)),
                        #     bigquery.ScalarQueryParameter("text_sale", "STRING", text_sale)
                        # ]

                        # job = bigquery_Client.query(query, job_config=job_config)
                        # job.result()


        # query = """
        # SELECT AVG(average_rating) 
        #     FROM `{}.CSM.csm_data`
        #     WHERE employee_name = @text_name;
        #  """.format(project_id)

        # job_config = bigquery.QueryJobConfig()
        # job_config.query_parameters = [
        #     bigquery.ScalarQueryParameter("predicted_topic_label", "STRING", predicted_topic_label),
        #     bigquery.ScalarQueryParameter("passage", "STRING", passage),
        #     bigquery.ScalarQueryParameter("response", "STRING", response),
        #     bigquery.ScalarQueryParameter("text_name", "STRING", text_name),
        #     bigquery.ScalarQueryParameter("text_rating", "STRING", rating_levels.index(rating_level)),
        #     bigquery.ScalarQueryParameter("text_sale", "STRING", text_sale)
        # ]

        # job = bigquery_Client.query(query, job_config=job_config)
        # job.result()

        # print(job.result)




        
        fig2, ax2 = plt.subplots()
        ax2.bar(rating_levels, rating_counts)
        ax2.set_xlabel('Rating Level')
        ax2.set_ylabel('Count')
        ax2.set_title('Salesman Rating Barometer')
        return fig1, fig2
    fig1, fig2 = index()
     
    fig1.savefig('static/figure1.png')
    fig2.savefig('static/figure2.png') 
    


    context = {
        "request": request,
        "video_path": video_path,
        "predicted_topic": predicted_topic_label,
        "passage":passage,
        "response":response,
        "text_name":text_name,
        "text_rating":text_rating,
        "text_sale":text_sale
    }


    return templates.TemplateResponse("result.html", context)









