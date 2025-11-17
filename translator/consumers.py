# translator/consumers.py
import json, base64
import numpy as np
import cv2
import mediapipe as mp
from google import genai

from channels.generic.websocket import AsyncWebsocketConsumer
from sympy import false
from tensorflow.keras.models import load_model
import asyncio
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tensorflow.keras.layers import LSTM, Dense
import math
from tensorflow.keras.models import Sequential

mp_holistic = mp.solutions.holistic

# Load model once
from django.conf import settings
import os

# -----------------------------------------
# FIX: Define actions BEFORE building the model
# -----------------------------------------
actions = np.array([
    'first', 'good', 'goodbye', 'hello', "I'm finished",
    'it was delicious', 'me', 'morning', 'shower', 'whitespace'
])

# -----------------------------------------
# Build LSTM model
# -----------------------------------------
hgfmodel = Sequential()
hgfmodel.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
hgfmodel.add(LSTM(128, return_sequences=True, activation='relu'))
hgfmodel.add(LSTM(64, return_sequences=False, activation='relu'))
hgfmodel.add(Dense(64, activation='relu'))
hgfmodel.add(Dense(32, activation='relu'))
hgfmodel.add(Dense(actions.shape[0], activation='softmax'))  # Now actions exists
hgfmodel.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hgfmodel.load_weights(os.path.join(settings.BASE_DIR, "Model", "action.h5"), by_name=True, skip_mismatch=True)
hgfmodel.summary()

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[1], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

CONF_THRESHOLD = 0.90

sequence = []
sentence = []
predictions = []
threshold = 0.9
numm = 1
frame_count = 0
useactionai = False
canbeaction = ["Mother", "G", "Me"]
sequencetrigger=True
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])







class ASLConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.running = True

        # Start the streaming task in background
        asyncio.create_task(self.stream_frames())

    async def disconnect(self, close_code):
        self.holistic.close()
        self.running = False

    async def receive(self, text_data):
        data = json.loads(text_data)
        msg_type = data.get("type")
        payload = data.get("data")
        client = genai.Client(api_key="AIzaSyBP6TAF27D-bIC1Fgg3UdvIlVrbDow_HkQ")

        sent = " ".join(sentence)

        if payload == "english":
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents="Translate these ASL Words into proper English. No explanation needed: " + sent
            )

            translated_text = response.text  # <-- correct way to extract

            await self.send(text_data=json.dumps({
                "type": "translated",
                "text": translated_text
            }))

        # if(payload=="english"):
        #     response = client.models.generate_content(
        #         model="gemini-2.0-flash-lite",
        #         contents="Translate these ASL Words into proper english. No explanation needed:" + sent
        #     )
        #     await self.send(text_data=json.dumps({
        #         "type":"translated",
        #         "text":('Translated'+str(response))
        #     }))

        if payload == "X":
            if sentence:  # prevent pop() on empty list
                sentence.pop()

            await self.send(text_data=json.dumps({
                "type": "translation_update",
                "translation": " ".join(sentence)
            }))
            pass

        if payload == "tagalog":
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=f"Translate these ASL words into proper Tagalog sentence. No explanation needed: {sent}"
            )

            translated_text = response.text  # EXTRACT CORRECT TEXT

            await self.send(text_data=json.dumps({
                "type": "translated",
                "text": translated_text
            }))


    async def stream_frames(self):
        os.makedirs('output_folder', exist_ok=True)
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        model_path = os.path.join(settings.BASE_DIR, "Model", "keras_model.h5")
        labels_path = os.path.join(settings.BASE_DIR, "Model", "labels.txt")
        classifier = Classifier(model_path, labels_path)
        offset = 20
        imgSize = 300
        labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "A", "B", "C", "D", "E", "F", "Father", "Food",
                  "G", "H", "I", "I Love You", "J", "K", "L", "Me", "Mother", "O", "P", "Play", "Q", "R", "S", "T", "U",
                  "V", "W", "X", "Y", "Z","M","N"]
        # Smoothing buffer
        prediction_buffer = []
        BUFFER_SIZE = 10  # change to 5, 10, 15 etc.

        while self.running:
            success, img = cap.read()
            #constantly keep tabs of frames
            img2 = img
            cv2.flip(img2, 1)
            image_rgb, results = mediapipe_detection(img2, self.holistic)
            draw_styled_landmarks(image_rgb, results)

            keypoints = extract_keypoints(results)
            global sequence, sentence, predictions, useactionai, sequencetrigger
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if((useactionai==True)):
                if(len(sequence)==30):
                    res = hgfmodel.predict(np.expand_dims(sequence, axis=0))[0]
                    image_rgb = prob_viz(res, actions, image_rgb, colors)

                _, buffer = cv2.imencode(".jpg", image_rgb)
                frame_bytes = buffer.tobytes()
                await self.send(bytes_data=frame_bytes)
                await asyncio.sleep(0)

                if(len(sequence)==30):

                    res = hgfmodel.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    # print("debugme")
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        # print("debugme1")
                        if res[np.argmax(res)] > threshold:
                            # print("debugme2")
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    if (actions[np.argmax(res)] != 'whitespace'):
                                        sequence = sequence[-1:]
                                        prediction_buffer = prediction_buffer[-1:]
                                        sentence.append(actions[np.argmax(res)])
                                        useactionai = False
                                        sequencetrigger = True
                                        await self.send(text_data=json.dumps({
                                            'translation': ' '.join(sentence)
                                        }))


                            else:
                                if (actions[np.argmax(res)] != 'whitespace'):
                                    print("aaaa")
                                    sequence = sequence[-1:]
                                    prediction_buffer = prediction_buffer[-1:]

                                    sentence.append(actions[np.argmax(res)])
                                    useactionai = False
                                    sequencetrigger = True
                                    await self.send(text_data=json.dumps({
                                        'translation': ' '.join(sentence)
                                    }))

            if(useactionai!=True):
                imgOutput = img.copy()
                hands, img = detector.findHands(img)
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                    # SAFE CROP SECTION — prevents crash
                    hImg, wImg = img.shape[:2]

                    x1 = max(0, x - offset)
                    y1 = max(0, y - offset)
                    x2 = min(wImg, x + w + offset)
                    y2 = min(hImg, y + h + offset)

                    imgCrop = img[y1:y2, x1:x2]

                    # Safety check
                    if imgCrop is None or imgCrop.size == 0:
                        print("Crop is empty, skipping frame")
                        _, buffer = cv2.imencode(".jpg", imgOutput)
                        frame_bytes = buffer.tobytes()
                        await self.send(bytes_data=frame_bytes)
                        continue

                    imgCropShape = imgCrop.shape
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        wCal = min(wCal, imgSize)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize

                        if (useactionai != True):
                            prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        else:
                            prediction = None
                            index = None
                        # Add to buffer
                        prediction_buffer.append(prediction)

                        # Keep buffer size fixed
                        if len(prediction_buffer) > BUFFER_SIZE:
                            prediction_buffer.pop(0)

                        # Only compute smoothed prediction when enough frames collected
                        if len(prediction_buffer) == BUFFER_SIZE:
                            # Average the confidence scores across frames
                            avg_confidence = np.mean(prediction_buffer, axis=0)


                            # Pick the highest-scoring label
                            smooth_index = int(np.argmax(avg_confidence))

                            # CONFIDENCE OF SMOOTHED OUTPUT (optional)
                            smooth_conf = avg_confidence[smooth_index]

                            # Print smooth result
                            print("Smoothed:", labels[smooth_index], "Confidence:", smooth_conf)

                            best_index = int(np.argmax(avg_confidence))
                            best_conf = float(avg_confidence[best_index])
                            best_label = labels[best_index]

                            if ((best_conf >= CONF_THRESHOLD)and(sequencetrigger==True)):
                                if(not sentence):
                                    prediction_buffer.clear()
                                    sentence.append(best_label)
                                elif(best_label != sentence[-1]):
                                    prediction_buffer.clear()
                                    sentence.append(best_label)
                            elif((len(prediction_buffer)>=BUFFER_SIZE)and(smooth_conf<0.3)):
                                #put action ai model here
                                if(sequencetrigger==True):
                                    print("weget")
                                    useactionai = True
                                    sequence = sequence[-1:]
                                    sequencetrigger=False
                            else:
                                print("Below threshold – skipped.")
                            # Clear buffer for next 5-frame batch
                            prediction_buffer.clear()

                            await self.send(text_data=json.dumps({
                                'translation': ' '.join(sentence)
                            }))
                            smooth_index = int(np.argmax(avg_confidence))  # CONFIDENCE OF SMOOTHED OUTPUT (optional)

                            # Use smoothed label for drawing
                            final_label = labels[smooth_index]
                        else:
                            # Not enough frames yet → fallback to instant prediction
                            final_label = labels[index]

                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        hCal = min(hCal, imgSize)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)

                        # Add to buffer
                        prediction_buffer.append(prediction)

                        # Keep buffer size fixed
                        if len(prediction_buffer) > BUFFER_SIZE:
                            prediction_buffer.pop(0)

                        # Only compute smoothed prediction when enough frames collected
                        if len(prediction_buffer) == BUFFER_SIZE:
                            # Average the confidence scores across frames
                            avg_confidence = np.mean(prediction_buffer, axis=0)
                            # Pick the highest-scoring label
                            smooth_index = int(np.argmax(avg_confidence))

                            # CONFIDENCE OF SMOOTHED OUTPUT (optional)
                            smooth_conf = avg_confidence[smooth_index]

                            # Print smooth result
                            print("Smoothed:", labels[smooth_index], "Confidence:", smooth_conf)

                            best_index = int(np.argmax(avg_confidence))
                            best_conf = float(avg_confidence[best_index])
                            best_label = labels[best_index]

                            if ((best_conf >= CONF_THRESHOLD)and(sequencetrigger==True)):
                                if(not sentence):
                                    prediction_buffer.clear()
                                    sentence.append(best_label)
                                elif(best_label != sentence[-1]):
                                    sentence.append(best_label)
                            elif ((len(prediction_buffer) >= BUFFER_SIZE) and (smooth_conf < 0.3)):
                                #put action ai model here
                                if(sequencetrigger==True):
                                    print("where")
                                    useactionai = True
                                    sequence = sequence[-1:]
                                    sequencetrigger=False
                            else:
                                print("Below threshold – skipped.")
                            # Clear buffer for next 5-frame batch
                            prediction_buffer.clear()

                            await self.send(text_data=json.dumps({
                                'translation': ' '.join(sentence)
                            }))
                            smooth_index = int(np.argmax(avg_confidence))  # CONFIDENCE OF SMOOTHED OUTPUT (optional)
                            # Use smoothed label for drawing
                            final_label = labels[smooth_index]
                        else:
                            # Not enough frames yet → fallback to instant prediction
                            final_label = labels[index]
                    # Safe bounding box limits (use where text will appear)
                    x1 = max(0, x - offset)
                    y1 = max(0, y - offset - 50)
                    x2 = min(imgOutput.shape[1], x - offset + 200)  # width for text overlay
                    y2 = min(imgOutput.shape[0], y - offset)  # height for text overlay

                    roi_h = y2 - y1
                    roi_w = x2 - x1

                    # Prevent negative dimensions
                    if roi_h <= 0 or roi_w <= 0:
                        print("Invalid ROI, skipping frame")
                        continue

                    text_img = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)

                    # Draw REGULAR text
                    cv2.putText(text_img, final_label, (10, roi_h - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

                    # Mirror the text
                    text_img_flip = cv2.flip(text_img, 1)

                    # Grab ROI from original image
                    roi = imgOutput[y1:y2, x1:x2]

                    # Resize overlay to ROI (fixes size mismatch error!)
                    text_img_flip = cv2.resize(text_img_flip, (roi.shape[1], roi.shape[0]))

                    # Blend overlay into image
                    imgOutput[y1:y2, x1:x2] = cv2.addWeighted(roi, 1, text_img_flip, 1, 0)

                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                  (x + w + offset, y + h + offset), (255, 0, 255), 4)

                _, buffer = cv2.imencode(".jpg", imgOutput)
                frame_bytes = buffer.tobytes()
                await self.send(bytes_data=frame_bytes)

                await asyncio.sleep(0)


def actionprocess():
    # --- Optional AI processing ---
    image_rgb=1
    print(len(sequence))
    if len(sequence) == 30:
        res = hgfmodel.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])
        predictions.append(np.argmax(res))
        if np.unique(predictions[-10:])[0] == np.argmax(res):
            if res[np.argmax(res)] > threshold:

                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        if (actions[np.argmax(res)] != 'whitespace'):
                            sentence.append(actions[np.argmax(res)])
                else:
                    if (actions[np.argmax(res)] != 'whitespace'):
                        sentence.append(actions[np.argmax(res)])

        frame = prob_viz(res, actions, image_rgb, colors)
        frame = cv2.flip(frame, 1)
    # --------------------------------

    # Encode frame to JPEG

