# Deep Learning: Predicting Income Levels with Keras and Scikit-learn

This project utilizes deep learning techniques to predict income levels based on the UCI Adult dataset. It demonstrates the integration of TensorFlow/Keras models within a Scikit-learn pipeline, encompassing data preprocessing, model training, evaluation, and visualization.

## 📂 Project Structure

```
Deep-Learning/
├── data_loader.py       # Handles data loading, cleaning, encoding, and splitting
├── keras_model.py       # Defines and compiles the Keras model wrapped with scikeras
├── main.py              # Executes the training pipeline and evaluation
├── requirements.txt     # Lists all Python dependencies
└── README.md            # Project documentation
```

## 📊 Dataset

The project uses the [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult), which contains demographic information to predict whether an individual's income exceeds $50K/year.

## ⚙️ Features

- **Data Preprocessing**: Implements one-hot encoding for categorical variables and MinMax scaling for continuous features.
- **Model Architecture**: Constructs a Keras Sequential model with two hidden layers using ReLU activation and a sigmoid output layer for binary classification.
- **Pipeline Integration**: Wraps the Keras model using `scikeras.wrappers.KerasClassifier` to integrate seamlessly with Scikit-learn's pipeline.
- **Evaluation Metrics**: Calculates ROC AUC score and plots the ROC curve for model performance assessment.

## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/MHasso/Deep-Learning.git
   cd Deep-Learning
   ```

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv ml-env
   source ml-env/bin/activate  # On Windows: ml-env\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the main script to train the model and evaluate its performance:

```bash
python main.py
```

This will:

- Load and preprocess the dataset.
- Train the Keras model within a Scikit-learn pipeline.
- Output the ROC AUC score.
- Display the ROC curve.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
