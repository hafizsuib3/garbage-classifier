# 🗑️ Garbage Classifier

A deep learning model that classifies garbage images into **6 categories**:  
`trash`, `plastic`, `paper`, `metal`, `glass`, `cardboard`  

Built with **TensorFlow / Keras** using **MobileNetV2 transfer learning** and fine-tuning.

---

## 📊 Dataset
- **Source:** Custom garbage dataset (`/images/` folder in Google Drive)  
- **Samples per class:** 300 (balanced dataset)  
- **Split:** 80% training / 20% validation  
- **Augmentation:** rotation, zoom, shift, shear, horizontal flip  

---

## 🧠 Model Architecture
- **Base Model:** MobileNetV2 (pretrained on ImageNet)  
- **Fine-tuned:** Top 50 layers unfrozen  
- **Classifier Head:**  
  - Global Average Pooling  
  - Dense(256, ReLU)  
  - Dropout(0.4)  
  - Dense(6, Softmax)  

Optimizer: **Adam (lr=1e-4)**  
Loss: **Categorical Crossentropy**  
Metrics: **Accuracy**

---

## 📈 Results
- **Validation Accuracy:** `XX.XX%`  
- **Macro F1-score:** `XX.XX`  

### Accuracy Curve
![Accuracy Curve](results/accuracy_curve.png)

### Loss Curve
![Loss Curve](results/loss_curve.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

---

## 🔍 Sample Prediction
Example of model prediction on validation data:

| Input Image | Predicted Class | Confidence |
|-------------|----------------|-------------|
| ![Sample](results/sample_prediction.png) | Plastic | 94% |

---

## ⚡ Challenges & Solutions
- **Challenge:** Dataset imbalance  
  ✅ Capped samples at 300/class and used augmentation  

- **Challenge:** Overfitting after ~5 epochs  
  ✅ Added Dropout + Early Stopping  

---

## 🚀 How to Run
Clone the repository:
```bash
git clone https://github.com/yourusername/garbage-classifier.git
cd garbage-classifier
