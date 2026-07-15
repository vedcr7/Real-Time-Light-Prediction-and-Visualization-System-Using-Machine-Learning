# 🌞 Real-Time Light State Prediction and Visualization Using Machine Learning and Arduino

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Arduino](https://img.shields.io/badge/Arduino-Uno-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random_Forest-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **B.Tech CSE (AI) Mini Project**  
> **Author:** Vedaansh Gupta

## 📖 Overview
An end-to-end IoT + Machine Learning project that acquires real-time ambient light data from an Arduino Uno and LDR sensor, engineers temporal features, predicts Light/Dark states using a Random Forest classifier, and visualizes predictions through a live Tkinter dashboard.

## ✨ Features
- Real-time Arduino serial communication
- LDR sensor data logging
- CSV dataset generation
- Feature engineering (Hour, Day, Month, Elapsed Time)
- Random Forest (100 estimators)
- Live Tkinter GUI
- Matplotlib scatter plot
- 93.20% prediction accuracy

## 🛠 Tech Stack
Python • Arduino • Scikit-Learn • Pandas • NumPy • Matplotlib • Tkinter • Serial

## 🏗 Architecture
```text
LDR Sensor
    │
Arduino Uno
    │
Serial Communication
    │
Python Data Logger
    │
Feature Engineering
    │
Random Forest
    │
Prediction
    │
Tkinter + Matplotlib Dashboard
```

## 🤖 Machine Learning Pipeline
1. Data Collection
2. CSV Logging
3. Timestamp Feature Extraction
4. Label Encoding
5. Train/Test Split (80/20)
6. Random Forest Training
7. Real-Time Prediction
8. Visualization

## 📊 Results

| Metric | Value |
|---|---:|
| Algorithm | Random Forest |
| Estimators | 100 |
| Accuracy | **93.20%** |
| Train/Test Split | 80/20 |

The model achieved excellent classification performance for Light and Dark states under varying lighting conditions.

## 📷 Project Gallery

### image1.jpg
![](images/image1.jpg)

### image10.png
![](images/image10.png)

### image11.png
![](images/image11.png)

### image12.png
![](images/image12.png)

### image13.jpeg
![](images/image13.jpeg)

### image2.png
![](images/image2.png)

### image3.png
![](images/image3.png)

### image4.png
![](images/image4.png)

### image5.jpeg
![](images/image5.jpeg)

### image6.png
![](images/image6.png)

### image7.png
![](images/image7.png)

### image8.png
![](images/image8.png)

### image9.png
![](images/image9.png)


## 🚀 Installation
```bash
git clone https://github.com/yourusername/Real-Time-Light-State-Prediction.git
cd Real-Time-Light-State-Prediction
pip install -r requirements.txt
python gui.py
```

## 📌 Future Improvements
- ESP32 Integration
- IoT Cloud Dashboard
- Raspberry Pi Deployment
- Mobile App
- Multiple Environmental Sensors
- Deep Learning Models

## 👨‍💻 Author
**Vedaansh Gupta**  
B.Tech Computer Science & Engineering (AI)  
Graphic Era (Deemed to be University)

If you found this project useful, consider giving it a ⭐.
