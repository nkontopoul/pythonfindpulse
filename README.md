# pythonfindpulse
This program captures video from your webcam, detects your face, and focuses on the regions next to your eyes (temples) to monitor pulse changes. It processes these regions to extract photoplethysmographic (PPG) signals, which are then used to estimate heart rate.
The heart rate is displayed in real-time on the video feed and plotted over time in a separate Matplotlib window.
Basic Components:

    Face and Temple Detection: Uses Haar cascades.
    PPG Signal Extraction: Monitors intensity changes in the temple regions.
    Heart Rate Calculation: Estimates heart rate from signal peaks.
    Real-Time Display: Shows heart rate on video and in a plot.
Steps:

    Detect face and regions next to the eyes (temples).
    Extract average color intensity from these regions.
    Filter the signal to isolate the heartbeat frequency range.
    Detect peaks in the filtered signal to estimate heart rate.
    Display the heart rate in real-time on the video feed and plots it over time in a separate window.

Requirements for the Code:

    Libraries: You need to install the following Python libraries:
        opencv-python
        numpy
        scipy
        matplotlib

    Install them using pip:

    pip install opencv-python numpy scipy matplotlib

    Haar Cascade Files: Ensure the following XML files are in the same directory as your script or provide the correct path:
        haarcascade_frontalface_default.xml
        haarcascade_eye.xml

    Python Version: Ensure you are using Python 3.x.

Steps to Run the Code:

    Place Haar Cascade Files: Download and place the XML files in the same directory as your script.
    Run the Script: Execute the script from the command line or an IDE.
