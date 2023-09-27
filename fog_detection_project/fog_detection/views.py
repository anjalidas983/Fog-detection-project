from django.shortcuts import render
import cv2
import numpy as np
from django.core.files.storage import FileSystemStorage
import os

# Create your views here.
def home(request):
    if request.method=="POST" and 'video_file' in request.FILES:
        videofile =request.FILES['video_file']
        file_storage=FileSystemStorage()
        filename=file_storage.save(videofile.name,videofile)
        # video_path =file_storage.url(filename)
        video_path =os.path.join(file_storage.location,filename)
        if not os.path.exists(video_path):
            error_message = "Video file does not exist."
            return render(request, 'detection_error.html', {'error': error_message})
        
        cap1= cv2.VideoCapture(video_path)
        if not cap1.isOpened():
            error_message="Error opening video file."
            return render(request,'detection_error.html',{'error':error_message})
        delay = 50  
        fog_percentages = []
        while cap1.isOpened():
            ret, frame = cap1.read()
            if not ret:
                # return render(request,'home.html')
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)
            diff = cv2.absdiff(gray_frame, blurred_frame)
            _, thresholded_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            white_pixel_count = np.sum(thresholded_diff == 255)
            
            frame_size = frame.shape[0] * frame.shape[1]
            fog_percentage = (white_pixel_count / frame_size) * 100.0
            fog_percentages.append(fog_percentage)
            cv2.putText(frame, f'Fog Percentage: {fog_percentage:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Fog Detection', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                # return render(request,'home.html')
                break

        cap1.release()
        cv2.destroyAllWindows()
        if not fog_percentages:
            error_message = "No frames processed"
            return render(request, 'detection_error.html', {'error': error_message})


        average_fog_percentage=(sum(fog_percentages)/len(fog_percentages))*100
        f = int(average_fog_percentage)
        return render(request,'fog_percentage.html',{'f':f})
    else:
        return render(request,'home.html')
    return render(request,'home.html')       