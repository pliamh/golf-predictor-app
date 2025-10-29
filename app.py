import os
import re
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_absolute_error  # optional

# -----------------------
# Optional OpenAI (lazy)
# -----------------------
@st.cache_resource(show_spinner=False)
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return None
    try:
        from openai import OpenAI  # lazy import
        return OpenAI(api_key=api_key)
    except Exception:
        return None

# -----------------------
# Page config / Styles
# -----------------------
st.set_page_config(
    page_title="⛳ Golf Score Predictor",
    page_icon="⛳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { padding: 1rem; }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 2rem;
        border-radius: 10px;
        border: 3px solid #4caf50;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-score {
        font-size: 3rem;
        color: #1b5e20;
        font-weight: bold;
    }
    h1 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Files / Constants
# -----------------------
MODEL_FILE = "golf_model.pkl"
HISTORY_FILE = "golf_history.csv"
NEW_ROUNDS_FILE = "new_rounds.csv"

# One source of truth for features
FEATURE_COLUMNS = [
    'course_rating', 'slope', 'is_home_course', 'month', 'day_of_week', 'is_weekend',
    'days_since_last_round', 'rolling_avg_5', 'rolling_std_5',
    'temp_mean', 'precipitation', 'wind_speed_max', 'is_hot', 'is_windy', 'has_rain',
    'difficulty_vs_home', 'pcc',
    'practice_score', 'greens_numeric', 'rough_numeric',
    'has_construction', 'has_standing_water', 'time_numeric',
    'num_partners', 'physical_condition', 'course_condition'  # <- new feature
]

# -----------------------
# Helpers
# -----------------------
def parse_partner_names(raw: str):
    if not raw:
        return []
    parts = re.split(r"[,\n;]+", raw)
    return [p.strip() for p in parts if p.strip()]

def backfill_missing_columns(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    """Ensure all keys in defaults exist as columns; fill if missing."""
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df

# -----------------------
# Data / Model utilities
# -----------------------
def train_model(df: pd.DataFrame):
    """Train and persist an XGBoost regressor using FEATURE_COLUMNS."""
    X = df[FEATURE_COLUMNS]
    y = df['score']
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def initialize_data():
    """Create initial history/model and new_rounds files on first run."""
    if not os.path.exists(HISTORY_FILE):
        ghin_data = {
            'date': ['2025-10-22', '2025-10-19', '2025-10-17', '2025-10-13', '2025-10-10',
                     '2025-10-08', '2025-10-06', '2025-10-03', '2025-09-29', '2025-09-22',
                     '2025-09-21', '2025-09-20', '2025-09-19', '2025-09-15', '2025-09-14',
                     '2025-09-13', '2025-09-12', '2025-09-08', '2025-09-03', '2025-08-19',
                     '2025-08-18', '2025-08-16', '2025-08-15', '2025-07-08', '2025-06-30',
                     '2025-06-29', '2025-06-23', '2025-06-20', '2025-06-18', '2025-06-17',
                     '2025-06-17', '2025-06-16', '2025-06-15', '2025-06-14', '2025-06-13',
                     '2025-06-11', '2025-06-08', '2025-06-07', '2025-06-06', '2025-06-05',
                     '2025-06-02', '2025-06-01', '2025-05-30', '2025-05-27', '2025-05-25',
                     '2025-05-24', '2025-05-20', '2025-05-17', '2025-05-16', '2025-05-13'],
            'score': [103, 107, 102, 103, 102, 95, 102, 102, 102, 92,
                      95, 98, 96, 102, 95, 99, 92, 102, 95, 104,
                      98, 103, 102, 90, 93, 88, 105, 93, 92, 97,
                      97, 97, 98, 103, 107, 90, 95, 95, 102, 95,
                      105, 96, 103, 96, 99, 89, 95, 97, 101, 112]
        }
        df = pd.DataFrame(ghin_data)
        df['date'] = pd.to_datetime(df['date'])

        # Baseline engineered features
        df['course_rating'] = 67.9
        df['slope'] = 120
        df['is_home_course'] = 1
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['pcc'] = 0

        temp_map = {5: 82, 6: 86, 7: 88, 8: 88, 9: 87, 10: 83}
        df['temp_mean'] = df['month'].map(temp_map)
        df['precipitation'] = 0
        df['wind_speed_max'] = 10
        df['is_hot'] = (df['temp_mean'] > 85).astype(int)
        df['is_windy'] = 0
        df['has_rain'] = 0
        df['difficulty_vs_home'] = 0

        df = df.sort_values('date')
        df['days_since_last_round'] = df['date'].diff().dt.days.fillna(7)
        df['rolling_avg_5'] = df['score'].rolling(window=5, min_periods=1).mean()
        df['rolling_std_5'] = df['score'].rolling(window=5, min_periods=1).std().fillna(0)

        # Qualitative defaults
        df['practice_score'] = 0
        df['greens_numeric'] = 1
        df['rough_numeric'] = 1
        df['has_construction'] = 0
        df['has_standing_water'] = 0
        df['time_numeric'] = 0
        df['num_partners'] = 3
        df['physical_condition'] = 7
        df['course_condition'] = 7  # include new feature from the start

        df.to_csv(HISTORY_FILE, index=False)
        train_model(df)

    if not os.path.exists(NEW_ROUNDS_FILE):
        pd.DataFrame(columns=['date', 'predicted_score', 'actual_score', 'partner_names']).to_csv(NEW_ROUNDS_FILE, index=False)

def migrate_schema_and_retrain_if_needed():
    """Backfill any missing columns and retrain if the model schema doesn't match FEATURE_COLUMNS."""
    # 1) Ensure history has all required feature columns
    df_hist = pd.read_csv(HISTORY_FILE)
    defaults = {
        'course_rating': 67.9,
        'slope': 120,
        'is_home_course': 1,
        'month': pd.to_datetime(df_hist['date']).dt.month if 'date' in df_hist.columns else 6,
        'day_of_week': 3,
        'is_weekend': 0,
        'days_since_last_round': 7,
        'rolling_avg_5': df_hist['score'].rolling(5, min_periods=1).mean() if 'score' in df_hist.columns else 95,
        'rolling_std_5': df_hist['score'].rolling(5, min_periods=1).std().fillna(0) if 'score' in df_hist.columns else 0,
        'temp_mean': 82,
        'precipitation': 0,
        'wind_speed_max': 10,
        'is_hot': 0,
        'is_windy': 0,
        'has_rain': 0,
        'difficulty_vs_home': 0,
        'pcc': 0,
        'practice_score': 0,
        'greens_numeric': 1,
        'rough_numeric': 1,
        'has_construction': 0,
        'has_standing_water': 0,
        'time_numeric': 0,
        'num_partners': 3,
        'physical_condition': 7,
        'course_condition': 7,
    }
    df_hist = backfill_missing_columns(df_hist, defaults)
    # Recompute rolling fields robustly if date/score exist
    if 'date' in df_hist.columns:
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist = df_hist.sort_values('date')
        if 'score' in df_hist.columns:
            df_hist['rolling_avg_5'] = df_hist['score'].rolling(5, min_periods=1).mean()
            df_hist['rolling_std_5'] = df_hist['score'].rolling(5, min_periods=1).std().fillna(0)
            df_hist['days_since_last_round'] = df_hist['date'].diff().dt.days.fillna(7)

    df_hist.to_csv(HISTORY_FILE, index=False)

    # 2) Try to use current model; if prediction on a single row fails, retrain
    model = load_model()
    need_retrain = False
    try:
        if model is None:
            need_retrain = True
        else:
            test_row = df_hist.iloc[[-1]][FEATURE_COLUMNS]
            _ = model.predict(test_row)  # if this raises, features
