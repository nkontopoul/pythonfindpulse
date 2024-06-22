import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def extract_temples_region(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # Define regions next to the eyes (temples)
        left_temple = frame[y+int(h*0.25):y+int(h*0.5), x:x+int(w*0.25)]
        right_temple = frame[y+int(h*0.25):y+int(h*0.5), x+int(w*0.75):x+w]
        return left_temple, right_temple
    return None, None

def extract_ppg_signal(region):
    mean_color = np.mean(region, axis=(0, 1))
    return mean_color

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')

    if not os.path.exists(face_cascade_path):
        print("Cascade file not found in the current directory.")
        return

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    cap = cv2.VideoCapture(0)
    intensity_values = []
    fs = 30  # Assuming 30 FPS
    heart_rates = []

    fig, ax = plt.subplots()
    ax.set_title('Heart Rate Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heart Rate (BPM)')
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame_num):
        if len(heart_rates) > 0:
            line.set_data(range(len(heart_rates)), heart_rates)
            ax.set_xlim(0, len(heart_rates))
            ax.set_ylim(0, max(heart_rates) + 10)
        return line,

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True)

    plt.ion()
    plt.show()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left_temple, right_temple = extract_temples_region(frame, face_cascade)
        if left_temple is not None and right_temple is not None:
            intensity_left = extract_ppg_signal(left_temple)
            intensity_right = extract_ppg_signal(right_temple)
            intensity = (intensity_left[1] + intensity_right[1]) / 2  # Average green channel
            intensity_values.append(intensity)

            if len(intensity_values) > fs * 10:  # At least 10 seconds of data
                filtered_signal = bandpass_filter(intensity_values[-fs*10:], 0.75, 3.0, fs)
                peaks, _ = find_peaks(filtered_signal, distance=fs/2)
                heart_rate = len(peaks) * 6  # 10-second window, multiply by 6 for BPM
                heart_rates.append(heart_rate)
                cv2.putText(frame, f"Heart Rate: {heart_rate:.2f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Left Temple', left_temple)
            cv2.imshow('Right Temple', right_temple)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
