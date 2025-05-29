# 🧭 Lab2 Positional ML

Welcome to the **Lab2 Positional ML** project!  
This repository provides a graphical interface and scripts for training, evaluating, and testing machine learning models for positional data using WiFi fingerprints.

## 🚀 Features

- **Train Mode**: Train your ML models on WiFi fingerprint data.
- **Evaluate Mode**: Evaluate trained models with various metrics.
- **Prediction Evaluation**: Upload your predictions and get instant feedback with regression and classification metrics.
- **User-Friendly GUI**: Simple interface built with Tkinter for easy interaction.

## 🗂️ Project Structure

```
.
├── main_gui.py                # Main graphical interface
├── train_mode.py              # Training logic
├── eval_mode.py               # Evaluation logic
├── eval_pred_mode.py          # Prediction evaluation logic
├── model_class.py             # Model class definitions
├── predictor.py               # Prediction utilities
├── metrics_eval.py            # Metrics computation
├── utils_data.py              # Data utilities
├── merged_fingerprints_wifi.csv # Example dataset
├── requirements.txt           # Python dependencies
├── Marcon_Lorenzo_labAvanzato2.pdf # Project documentation
├── pdf_merger/                # PDF utility scripts
├── results_1507/              # Results and outputs
└── uniud_logo.jpg             # University logo
```

## 🖥️ How to Run

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Start the GUI**  
   ```
   python main_gui.py
   ```

3. **Use the interface**  
   - Click **TRAIN** to train a model.
   - Click **EVALUATE** to evaluate a model.
   - Click **EVALUATE PREDICTION** to upload your predictions and see metrics.

## 📊 Metrics Supported

- RMSE, MAE, MSE, R², Explained Variance
- Accuracy, Precision, Recall, F1 Score
- Median Absolute Error, Mean Error, 2D Error

## 📄 Documentation

See [Marcon_Lorenzo_labAvanzato2.pdf](Marcon_Lorenzo_labAvanzato2.pdf) for detailed project information.

## 🏫 Credits

Developed for the Advanced Database Laboratory at the University of Udine.

---

*Happy experimenting! 🚦*