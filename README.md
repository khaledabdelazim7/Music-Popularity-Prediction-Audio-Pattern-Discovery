# 🎵 Music Popularity Prediction & Audio Pattern Discovery  
### Advanced Machine Learning & Deep Learning on 114,000 Spotify Tracks

This project develops a full end-to-end pipeline for predicting music popularity and discovering hidden audio patterns using the **Spotify Tracks Dataset**.  
It integrates **data preprocessing**, **supervised learning**, **unsupervised clustering**, **deep learning**, and an interactive **xg boost app**.

-----

## 📂 Dataset  
**Dataset:** Spotify Tracks Dataset
- **114,000+ tracks**  
- **18 audio features**  
- **Target:** `popularity` (0–100)

[Open the Spotify Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

Key features include:

- `danceability`, `energy`, `loudness`, `speechiness`,  
- `acousticness`, `instrumentalness`, `liveness`, `valence`,  
- `tempo`, `duration_ms`, `artist_name`, `genre`, and more.

---

## 🧹 1. Data Preprocessing  

### **User Story — Data Engineer @ Kkaled Abdelazim**  
> *"I want to process 114,000 tracks so that we build accurate popularity models."*

### ✔ Requirements Implemented
- Genre-based imputation for missing values  
- Derived features:
  - `energy_dance_ratio`
  - `acoustic_electronic_spectrum`
- Log-transform of `duration_ms`  
- Encoding:
  - Artist frequency/target encoding  
  - Genre embeddings  
- Popularity-balanced sampling  
- Scaling using StandardScaler  

---

## 🔍 2. Exploratory Data Analysis (EDA)

### **User Story — Music Analyst**  
> *"I want to understand popularity drivers and emerging trends."*

### ✔ Requirements Implemented
- Popularity distribution by decade & genre  
- Correlation matrices and pair plots  
- Temporal trends across decades  
- Artist-level popularity trajectory analysis  
- Genre evolution & crossover analysis  

---

## 🤖 3. Supervised Machine Learning

### **User Story — Data Scientist @ Mohamed Sheref**  
> *"I want to predict song popularity using ensemble models."*

### ✔ Models Implemented
- **XGBoost Regressor** + Bayesian Optimization  
- **Random Forest Regressor** + Feature Importance  
- **Neural Network with Embeddings**  

### ✔ Evaluation Metrics
- RMSE / MAE  
- High-popularity recall  
- Genre-specific performance  
- Temporal cross-validation (avoids decade leakage)

---

## 🎧 4. Unsupervised Machine Learning (Clustering & Discovery)

### **User Story — Music Curator @ Mohamed Mostafa**  
> *"I want to discover natural music clusters beyond traditional genres."*

### ✔ Techniques
- **KMeans (k=12)** micro-genre discovery  
- **DBSCAN** for outlier/niche music detection  
- **Hierarchical clustering** taxonomy  
- **PCA & t-SNE** for cluster visualization  
- Cluster validation using musical traits  

---

## 🧠 5. Deep Learning

### **User Story — AI Researcher @ Moaaz Hail**  
> *"I want deep learning models that capture complex audio patterns."*

### ✔ Implemented Models
- **DNN Regressor**
  - 5 hidden layers  
  - BatchNorm + Dropout  
  - Predicts popularity  
- **Autoencoder**
  - Unsupervised representation learning  
  - Latent embeddings used as extra features  
- **Cluster Embedding Integration**
- **Attention Mechanisms**
  - Interpret feature contribution  
- Benchmark vs traditional ML methods  

---

## 🖥 6. Deployment

### **User Story — Product Manager @ Kkaled Abdelazim**
> *"I want an interactive music analysis platform for 114,000+ tracks."*

### ✔ Streamlit Features
- **Real-time popularity prediction**  
- **Cluster explorer** with similar track recommendations  
- Genre evolution visualization  
- Artist similarity & trend analysis  
- Playlist generator using audio preferences  

---

## 🚀 Installation

```bash
git clone <https://github.com/khaledabdelazim7/Music-Popularity-Prediction-Audio-Pattern-Discovery>

