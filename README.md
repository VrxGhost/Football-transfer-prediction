# ⚽ Football Transfer Value Prediction

A machine learning project that predicts football player transfer market values using real Transfermarkt data.

---

## 🎯 Problem Statement

Football clubs spend billions on player transfers every year. Accurately estimating a player's market value helps clubs make smarter decisions. This project builds a regression model that predicts a player's transfer value (in €) based on their performance stats, demographics, and contract situation.

---
## 📌 About the Project
 
Football transfers are one of the most financially significant events in professional sports. This project uses machine learning to analyze player attributes — such as age, market value, performance stats, and contract details — to predict the likelihood of a player being transferred.
 
The model is trained and evaluated on a Kaggle dataset and follows a complete ML pipeline: data cleaning → feature engineering → model training → evaluation.
 
🔗 **Kaggle Notebook:** [Football Transfer Prediction by duttaarya](https://www.kaggle.com/code/duttaarya/football-transfer-prediction)
 
---
 
## 🧠 How It Works
 
The model pipeline includes:
 
- **Data Preprocessing** – handling missing values, encoding categorical features, scaling
- **Exploratory Data Analysis (EDA)** – understanding patterns in player stats and transfer history
- **Feature Engineering** – selecting the most relevant features influencing transfer decisions
- **Model Training** – training classification models (e.g., Random Forest, Logistic Regression, etc.)
- **Evaluation** – measuring accuracy, precision, recall, and F1-score
---

## 📊 Dataset

**Source:** [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) — Kaggle

Files used:
- `players.csv` — player demographics (age, position, height, nationality)
- `player_valuations.csv` — historical market values per player
- `appearances.csv` — match-level performance data (goals, assists, minutes)

Dataset size: ~450,000 valuation records, ~31,000 unique players

---

## 🔧 Features Used

| Feature | Description |
|---|---|
| `age` | Player age (derived from date_of_birth) |
| `height_in_cm` | Player height |
| `position` | Playing position (encoded) |
| `foot` | Dominant foot (encoded) |
| `contract_years_left` | Years remaining on contract |
| `goals_per_match` | Career goals divided by matches played |
| `assists_per_match` | Career assists divided by matches played |
| `matches_played` | Total career appearances |

**Target variable:** `market_value_in_eur` (latest valuation per player)

---

## 🏗️ Project Workflow

```
1. Data Loading       → Load players, valuations, appearances CSVs
2. Data Merging       → Join tables on player_id
3. Feature Engineering→ Create age, contract_years_left, performance ratios
4. Data Cleaning      → Handle missing values, remove leakage
5. Encoding           → One-hot encode position and foot
6. Baseline Model     → Linear Regression (R² ~ 0.07)
7. Feature Upgrade    → Add contract + performance features (R² ~ 0.25)
8. Model Upgrade      → Random Forest Regressor (R² ~ 0.60)
9. Overfitting Control→ Tune max_depth, min_samples_leaf
```

---

## 📈 Results

| Model | Features | Test R² |
|---|---|---|
| Linear Regression | Age + Height | 0.07 |
| Linear Regression | + Position + Foot | 0.07 |
| Linear Regression | + Contract Years | 0.18 |
| Linear Regression | + Performance Stats | 0.25 |
| Random Forest | All Features | 0.58 |
| **XGBoost** | **All Features** | **0.64** |

> XGboost significantly outperformed Random Forest by capturing non-linear relationships — e.g. young players with high goals-per-match have exponentially higher values.

---

## 🧠 Key Learnings

- **Data leakage** — accidentally including `market_value` inside features caused R² = 1.0 (false result). Identifying and fixing this was a critical ML debugging step.
- **Feature engineering matters more than model choice** — adding contract length and performance ratios improved R² from 0.07 → 0.25 before any model upgrade.
- **Log transformation** did not help here because Random Forest handles skewed targets natively.
- **Overfitting control** — limiting `max_depth` reduced train/test gap from 0.35 to 0.16.

---

## 🗂️ Project Structure

```
football-transfer-value/
│
├── data/
│   └── (download from Kaggle link above)
│
├── notebooks/
│   └── transfer_value_prediction.ipynb
│
├── model/
│   └── all_model.pkl
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/football-transfer-value.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download dataset from [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) and place CSVs in `data/`

4. Open and run the notebook:
```bash
jupyter notebook notebooks/transfer_value_prediction.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

---

## 👤 Author

**Arya Dutta**  
B.Tech Student  | 2nd year | Aspiring AI/ML Engineer
[Email Me](mailto:aryadutta1101@yahoo.com)
