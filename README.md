# CardioRiskAI  

**AI-powered Heart Disease Risk Predictor 🫀**  
This project uses a **TensorFlow-based ML model** to assess the risk of cardiovascular disease. It demonstrates end-to-end data preprocessing, training, evaluation, and model export.

---

## 🚀 Features  

- 🧠 **ML Model (TensorFlow/Keras)** – Deep learning model for heart disease prediction  
- 🔬 **Training Project (Python)** – Includes dataset, preprocessing, and training code  
- 📊 **Risk Analysis** – Predicts likelihood of cardiovascular disease  
- 💾 **Export Support** – Save models as `.h5` or convert to TensorFlow Lite (`.tflite`)  

---

## 📂 Repository Structure  

```
.
├── scripts/                          # Training scripts (Python, TensorFlow/Keras)
├── heart.csv                         # Dataset (UCI Heart Disease dataset)
├── metadata_writer_for_image_classifier.py
├── model/                            # Trained model files (.h5, .tflite)
├── .gitignore
└── README.md
```

---

## ⚙️ Usage  

1. Run training:  
```bash
python scripts/train.py
```

2. Export the trained model to `.h5` or `.tflite` for further use.

---

## 📊 Dataset  

We use the Heart Disease dataset (commonly from the UCI repository).

- Contains clinical parameters like age, cholesterol, resting ECG, max heart rate, etc.  
- `heart.csv` is included for reproducibility.  

---

## 🧠 Model  

- Framework: TensorFlow / Keras  
- Exported formats: `.h5` and `.tflite`  
- Evaluation metric: Accuracy (target ~85%)  
- Output classes:  
  - 0 → Low Risk  
  - 1 → High Risk  

---

## 🔒 Disclaimer  

This project is for educational and research purposes only. It should not be used as a replacement for professional medical diagnosis. Always consult a healthcare professional for medical advice.

---

## 🤝 Contributing  

- Fork the repo  
- Create a feature branch  
- Submit a Pull Request  

---

## 📜 License  

This project is licensed under the MIT License. 
