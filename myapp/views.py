from django.shortcuts import render
import os
from django.http import JsonResponse, HttpResponseBadRequest
from django.conf import settings
from ultralytics import YOLO
import cv2
import numpy as np


def detect(request):
    try:
        analysis_type = request.POST.get("type", "eye")  # eye OR face

        # -------------------- Select Model --------------------
        if analysis_type == "eye":
            model_path = os.path.join(settings.BASE_DIR, "myapp", "models", "best_eye.pt")
            detector_mode = "eye"
        else:
            model_path = os.path.join(settings.BASE_DIR, "myapp", "models", "best_face_float32.tflite")
            detector_mode = "face"

        model = YOLO(model_path)

        # -------------------- Validate File Upload --------------------
        if "image" not in request.FILES:
            return JsonResponse({"status": "0", "msg": "No image uploaded"})

        file = request.FILES["image"]

        # Save image temporarily
        temp_path = os.path.join(settings.BASE_DIR, file.name)
        with open(temp_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Load image
        image = cv2.imread(temp_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        total_abnormal = 0
        total_normal = 0

        # -------------------- EYE MODE --------------------
        if detector_mode == "eye":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

            if len(eyes) == 0:
                os.remove(temp_path)
                return JsonResponse({"status": "0", "msg": "No eye detected"})

            for (x, y, w, h) in eyes:
                crop = image[y:y + h, x:x + w]
                crop = cv2.resize(crop, (512, 512))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                results = model.predict(crop)
                probs = results[0].probs.data

                total_abnormal += probs[0].item()
                total_normal += probs[1].item()

            avg_abnormal = total_abnormal / len(eyes)
            avg_normal = total_normal / len(eyes)

        # -------------------- FACE MODE --------------------
        else:
    # Load as classification model
          model = YOLO(model_path, task="classify")

          resized = cv2.resize(rgb, (512, 512))

          results = model.predict(resized)

    # Extract classification probabilities
          probs = results[0].probs.data  # now works because task=classify

          avg_abnormal = float(probs[0])
          avg_normal = float(probs[1])


        # Remove temp file
        os.remove(temp_path)

        # -------------------- Confidence Result --------------------
        confidence = round(max(avg_abnormal, avg_normal) * 100, 2)

        if avg_abnormal > avg_normal:
            result = "Abnormal"
        else:
            result = "Normal"

        return JsonResponse({
            "status": "1",
            "msg": result,
            "confidence": confidence
        })

    except Exception as e:
        return HttpResponseBadRequest(str(e))
    analysis_type = request.POST.get("type", "eye")   # eye OR face

    # Select YOLO model
    if analysis_type == "eye":
        model_path = os.path.join(settings.BASE_DIR, "myapp", "models", "best_eye.pt")
        detector_mode = "eye"
    else:
        model_path = os.path.join(settings.BASE_DIR, "myapp", "models", "best_face_float32.tflite")
        detector_mode = "face"

    model = YOLO(model_path)

    # Validate image upload
    if "image" not in request.FILES:
        return JsonResponse({"status": "0", "msg": "No image uploaded"})

    file = request.FILES["image"]

    # Save temporarily
    temp_path = os.path.join(settings.BASE_DIR, file.name)
    with open(temp_path, "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    # Load image
    image = cv2.imread(temp_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    total_abnormal = 0
    total_normal = 0

    # ----------- EYE MODE (needs multiple crop detection) -----------
    if detector_mode == "eye":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)

        if len(eyes) == 0:
            os.remove(temp_path)
            return JsonResponse({"status": "0", "msg": "No eye detected"})

        # Predict for each detected eye crop
        for (x, y, w, h) in eyes:
            crop = image[y:y + h, x:x + w]
            crop = cv2.resize(crop, (512, 512))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            results = model.predict(crop)
            probs = results[0].probs.data

            total_abnormal += probs[0].item()
            total_normal += probs[1].item()

        avg_abnormal = total_abnormal / len(eyes)
        avg_normal = total_normal / len(eyes)

    # ----------- FACE MODE (NO CASCADE — FULL IMAGE) -----------
    else:
        resized = cv2.resize(rgb, (512, 512))
        results = model.predict(resized)
        probs = results[0].probs.data

        avg_abnormal = probs[0].item()
        avg_normal = probs[1].item()

    # Cleanup
    os.remove(temp_path)

    # Calculate confidence
    confidence = round(max(avg_abnormal, avg_normal) * 100, 2)

    # Return JSON
    if avg_abnormal > avg_normal:
        return JsonResponse({
            "status": "1",
            "msg": "Abnormal",
            "confidence": confidence
        })
    else:
        return JsonResponse({
            "status": "1",
            "msg": "Normal",
            "confidence": confidence
        })
    analysis_type = request.POST.get("type", "eye")   # <— read type from frontend

    # Select model based on analysis type
    if analysis_type == "eye":
        model_path = os.path.join(settings.BASE_DIR, "myapp", "models", "eye.pt")
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    else:
        model_path = os.path.join(settings.BASE_DIR, "myapp", "models", "face.pt")
        # cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"

    # Load YOLO model dynamically
    model = YOLO(model_path)

    # Validate image uploaded
    if "image" not in request.FILES:
        return JsonResponse({"status": "0", "msg": "No image uploaded"})

    file = request.FILES["image"]

    # Save to temp file
    temp_path = os.path.join(settings.BASE_DIR, file.name)
    with open(temp_path, "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    # Read image
    image = cv2.imread(temp_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load cascade classifier (eye or face)
    detector = cv2.CascadeClassifier(cascade_path)
    regions = detector.detectMultiScale(gray_image, 1.3, 5)

    if len(regions) == 0:
        os.remove(temp_path)
        return JsonResponse({"status": "0", "msg": "No eye/face detected. Try another photo."})

    total_abnormal = 0
    total_normal = 0

    # Loop through detected areas (eye or face)
    for (x, y, w, h) in regions:
        cropped = image[y:y + h, x:x + w]
        cropped = cv2.resize(cropped, (512, 512))
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # YOLO prediction
        results = model.predict(cropped_rgb)
        probs = results[0].probs.data

        total_abnormal += probs[0].item()
        total_normal += probs[1].item()

    avg_abnormal = total_abnormal / len(regions)
    avg_normal = total_normal / len(regions)

    os.remove(temp_path)

    confidence = round(max(avg_abnormal, avg_normal) * 100, 2)

    if avg_abnormal > avg_normal:
        return JsonResponse({
            "status": "1",
            "msg": "Abnormal",
            "confidence": confidence
        })
    else:
        return JsonResponse({
            "status": "1",
            "msg": "Normal",
            "confidence": confidence
        })
    # ---- Load Eye Model Only ----
    eye_model_path = os.path.join(settings.BASE_DIR, "myapp", "models", "eye.pt")
    model = YOLO(eye_model_path)

    # ---- Validate File Upload ----
    if "image" not in request.FILES:
        return JsonResponse({"status": "0", "msg": "No image uploaded"})

    file = request.FILES["image"]

    # ---- Save temporarily ----
    temp_path = os.path.join(settings.BASE_DIR, file.name)
    with open(temp_path, "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    # ---- Load Image ----
    image = cv2.imread(temp_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Eye detection using Haar Cascade
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)

    if len(eyes) == 0:
        os.remove(temp_path)
        return JsonResponse({"status": "0", "msg": "Use another photo"})

    total_abnormal = 0
    total_normal = 0

    for (ex, ey, ew, eh) in eyes:
        cropped_eye = image[ey:ey + eh, ex:ex + ew]
        cropped_eye = cv2.resize(cropped_eye, (512, 512))
        cropped_eye_rgb = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2RGB)

        # YOLO prediction
        results = model.predict(cropped_eye_rgb)
        probs = results[0].probs.data

        total_abnormal += probs[0].item()
        total_normal += probs[1].item()

    avg_abnormal = total_abnormal / len(eyes)
    avg_normal = total_normal / len(eyes)

    os.remove(temp_path)

    # Compute confidence score
    confidence = max(avg_abnormal, avg_normal) * 100

    if avg_abnormal > avg_normal:
        return JsonResponse({
            "status": "1",
            "msg": "Abnormal",
            "confidence": round(confidence, 2)
        })
    else:
        return JsonResponse({
            "status": "1",
            "msg": "Normal",
            "confidence": round(confidence, 2)
        })


def homepage(request):
    if request.method == "POST":
        try:
            return detect(request)
        except Exception as e:
            return HttpResponseBadRequest(str(e))

    return render(request, "index.html")
