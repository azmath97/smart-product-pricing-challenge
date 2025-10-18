# 🧠 Smart Product Pricing Challenge – ML Challenge 2025

## 📄 Problem Statement
In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction.  
This project develops a Machine Learning model that predicts the price of a product based on textual and visual product details such as brand, specifications, and quantity.  
The goal is to capture complex pricing relationships and suggest fair, data-driven price estimates.

---

## 📊 Dataset Description
- **Train set:** 75,000 products with detailed text descriptions, image links, and target price.  
- **Test set:** 75,000 products (without price labels) for evaluation.  
- **Columns:**
  - `sample_id`: Unique product ID  
  - `catalog_content`: Concatenated title, description, and Item Pack Quantity (IPQ)  
  - `image_link`: Public image URL  
  - `price`: Target variable (only in training data)

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Cleaned and normalized the text (`lowercase`, punctuation & stopword removal)
- Extracted **brand**, **specifications**, and **quantity keywords**
- Used **TF-IDF vectorization** to represent textual features
- Handled missing or malformed entries
- Scaled numeric features where applicable

### 2. Model Architecture
- Baseline models: **Linear Regression**, **Ridge Regression**, and **Random Forest Regressor**
- Final selected model: **Random Forest Regressor**
  - `n_estimators=300`, `max_depth=20`
  - Balanced bias–variance trade-off
- Image features were optionally extracted using a **ResNet50 pretrained model** (on selected runs)
- Combined text embeddings + image embeddings (where available)

### 3. Training & Validation
- Split training data into 80/20 (train/validation)
- Tuned hyperparameters using Grid Search
- Evaluated with **SMAPE (Symmetric Mean Absolute Percentage Error)**

---

## 🧮 Evaluation Metric
**SMAPE = (1/n) × Σ |pred - actual| / ((|pred| + |actual|) / 2)**  
Lower SMAPE indicates better accuracy.

**Model performance:**  
`SMAPE = 52%` on validation data

---

## 📈 Output Format
Final submission: `test_out.csv`

| sample_id | price |
|------------|--------|
| 12345      | 249.50 |
| 67890      | 179.99 |

Ensure all `sample_id` values from `test.csv` are present in the output.

---

## 🧩 Tools & Libraries
- Python 3.10  
- Pandas, NumPy, Scikit-learn  
- NLTK / SpaCy (for text cleaning)  
- TensorFlow / PyTorch (for optional image embeddings)  
- Matplotlib, Seaborn (visualization)

---

## 🚀 How to Run
1. Clone or unzip the repository  
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
