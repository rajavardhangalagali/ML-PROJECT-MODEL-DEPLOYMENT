# MoodMeal 🍽️
**Fine Dining Intelligence | AIML Subject Mini-Project**

MoodMeal is an end-to-end Full-Stack Data Science application that recommends the perfect meal based on a user's current state (mood, time of day, weather) and personal profile (hunger level, age group). It is powered by a **Random Forest Classifier** and wrapped in a beautiful, responsive, luxury-themed UI built with **Streamlit**.

## ✨ Features
- **Smart Recommendations:** Uses Machine Learning to predict one of seven detailed meal courses (e.g., Light Breakfast, Fast Food, Heavy Dinner, Comfort Food).
- **Confidence Breakdown:** Displays the model's confidence across all meal categories using dynamic matplotlib bar charts.
- **Model Analytics:** A dedicated tab showing dataset metrics, meal distribution pie charts, and real-time Feature Importance.
- **Responsive Luxury UI:** Injected custom CSS to strip away default Streamlit boilerplate, replacing it with a mobile-friendly, editorial-style layout.

## 🛠️ Tech Stack
- **Frontend / UI:** Streamlit, HTML/CSS, JavaScript (for smooth auto-scrolling)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `LabelEncoder`, `train_test_split`)
- **Data Visualization:** Matplotlib

## 🚀 How to Run Locally

1. **Ensure Python is installed**
   Make sure you have Python 3.8+ installed on your system.

2. **Install dependencies**
   Navigate to the project folder in your terminal and install the required libraries using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   Start the local development server by executing:
   ```bash
   streamlit run moodmeal_app_v2.py
   ```

4. **View in Browser**
   The app will automatically open in your default browser at `http://localhost:8501`.

## 📂 Project Structure
- `moodmeal_app_v2.py` - The main application script (contains UI, ML training, and inference pipeline).
- `dataset.csv` - The custom dataset (100 synthetic profiles) used to train the model.
- `requirements.txt` - Required Python packages.

## 👨‍💻 The Team
Engineered by the Next Generation for the Artificial Intelligence & Machine Learning (AIML) curriculum:
- **Rajavardhan S G** - Project Lead / Developer
- **Darshan G K** - Co-Developer / Designer
