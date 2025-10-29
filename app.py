import os
import re
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# -----------------------
# Optional OpenAI (lazy) — not used yet, safe to keep
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

# Single source of truth for feature names:
FEATURES = [
    'course_rating', 'slope', 'is_home_course', 'month', 'day_of_week', 'is_weekend',
    'days_since_last_round', 'rolling_avg_5', 'rolling_std_5',
    'temp_mean', 'precipitation', 'wind_speed_max', 'is_hot', 'is_windy', 'has_rain',
    'difficulty_vs_home', 'pcc',
    'practice_score', 'greens_numeric', 'rough_numeric',
    'has_construction', 'has_standing_water', 'time_numeric',
    'num_partners', 'physical_condition', 'course_condition'
]

# -----------------------
# Helpers
# -----------------------
def parse_partner_names(raw: str):
    """Return list of partner names from comma/semicolon/newline-separated string."""
    if not raw:
        return []
    parts = re.split(r"[,\n;]+", raw)
    return [p.strip() for p in parts if p.strip()]

# -----------------------
# Data / Model utilities
# -----------------------
def train_model(df: pd.DataFrame):
    """Train and persist an XGBoost regressor on the FEATURES list."""
    X = df[FEATURES]
    y = df['score']
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    return model

@st.cache_resource(show_spinner=False)
def load_model_raw():
    """Just load from disk if present."""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def ensure_new_rounds_file():
    if not os.path.exists(NEW_ROUNDS_FILE):
        pd.DataFrame(columns=['date', 'predicted_score', 'actual_score', 'partner_names']).to_csv(NEW_ROUNDS_FILE, index=False)

def initialize_data():
    """Create history/model/new_rounds files on first run."""
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

        # baseline features
        df['course_rating'] = 67.9
        df['slope'] = 120
        df['is_home_course'] = 1
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['pcc'] = 0

        # simple weather proxies by month
        temp_map = {5: 82, 6: 86, 7: 88, 8: 88, 9: 87, 10: 83}
        df['temp_mean'] = df['month'].map(temp_map).fillna(80)
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

        # qualitative defaults
        df['practice_score'] = 0
        df['greens_numeric'] = 1
        df['rough_numeric'] = 1
        df['has_construction'] = 0
        df['has_standing_water'] = 0
        df['time_numeric'] = 0
        df['num_partners'] = 3
        df['physical_condition'] = 7
        df['course_condition'] = 7

        df.to_csv(HISTORY_FILE, index=False)
        train_model(df)

    ensure_new_rounds_file()

def get_weather_forecast():
    """Very simple seasonal estimate placeholder."""
    month = datetime.now().month
    temp_map = {1: 68, 2: 70, 3: 74, 4: 78, 5: 82, 6: 86, 7: 88, 8: 88, 9: 87, 10: 83, 11: 76, 12: 70}
    temp = temp_map.get(month, 80)
    return {'temp_mean': temp, 'precipitation': 0, 'wind_speed_max': 10}

def load_or_retrain_model_if_needed():
    """
    Load the model; if its expected feature names don't match the current FEATURES,
    retrain on the current history to keep things compatible.
    """
    model = load_model_raw()
    if model is None:
        # Shouldn't happen after initialize_data(), but guard anyway
        df_hist = pd.read_csv(HISTORY_FILE)
        return train_model(df_hist)

    # Build a minimal test row with all current FEATURES to validate
    test_row = pd.DataFrame([ {f: 0 for f in FEATURES} ])
    try:
        _ = model.predict(test_row)
        return model  # OK
    except Exception:
        # Mismatch (likely old model without new features). Retrain.
        df_hist = pd.read_csv(HISTORY_FILE)
        # Ensure df_hist has all needed columns; fill any missing with defaults
        for f in FEATURES:
            if f not in df_hist.columns:
                df_hist[f] = 0
        return train_model(df_hist)

def make_prediction(inputs: dict):
    """Generate a score prediction and return (prediction, features, weather)."""
    model = load_or_retrain_model_if_needed()

    df_history = pd.read_csv(HISTORY_FILE)
    df_history['date'] = pd.to_datetime(df_history['date'])

    recent_scores = df_history.tail(5)['score']
    rolling_avg = recent_scores.mean()
    rolling_std = recent_scores.std() if len(recent_scores) > 1 else 0

    last_date = df_history['date'].max()
    days_since = (datetime.now() - last_date).days

    weather = get_weather_forecast()

    features = {
        'course_rating': 67.9 - (inputs['tee_box'] - 4) * 0.3,
        'slope': 120 - (inputs['tee_box'] - 4) * 2,
        'is_home_course': 1,
        'month': datetime.now().month,
        'day_of_week': datetime.now().weekday(),
        'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
        'days_since_last_round': days_since,
        'rolling_avg_5': rolling_avg,
        'rolling_std_5': rolling_std,
        'temp_mean': weather['temp_mean'],
        'precipitation': weather['precipitation'],
        'wind_speed_max': weather['wind_speed_max'],
        'is_hot': 1 if weather['temp_mean'] > 85 else 0,
        'is_windy': 1 if weather['wind_speed_max'] > 15 else 0,
        'has_rain': 1 if weather['precipitation'] > 0 else 0,
        'difficulty_vs_home': (120 - (inputs['tee_box'] - 4) * 2) - 120,
        'pcc': 0,
        'practice_score': inputs['practice_score'],
        'greens_numeric': inputs['greens_numeric'],
        'rough_numeric': inputs['rough_numeric'],
        'has_construction': inputs['has_construction'],
        'has_standing_water': inputs['has_standing_water'],
        'time_numeric': inputs['time_numeric'],
        'num_partners': inputs['num_partners'],
        'physical_condition': inputs['physical_condition'],
        'course_condition': inputs['course_condition']
    }

    X = pd.DataFrame([features], columns=FEATURES)
    prediction = float(model.predict(X)[0])
    return prediction, features, weather

# -----------------------
# Initialize
# -----------------------
initialize_data()

# -----------------------
# UI
# -----------------------
st.title("⛳ Golf Score Predictor")
st.markdown("### Predict your score")

tab1, tab2, tab3 = st.tabs(["🎯 Predict Score", "📝 Enter Actual Score", "📊 Stats"])

# TAB 1: Predict Score
with tab1:
    st.markdown("#### Today's Round")

    col1, col2 = st.columns(2)
    with col1:
        tee_box = st.selectbox("Tee Box", options=[1, 2, 3, 4, 5, 6], index=3, help="1 = hardest, 6 = easiest")
        practiced = st.selectbox("Practiced Today?", options=["No", "Driving Range", "Putting Green", "Both"])
        greens_speed = st.selectbox("Greens Speed", options=["Slow", "Medium", "Fast"], index=1)
        rough_thickness = st.selectbox("Rough Thickness", options=["Thin", "Normal", "Thick"], index=1)
    with col2:
        physical_condition = st.slider("Your Physical Condition", 1, 10, 7, help="How do you feel today?")
        course_condition = st.slider("Course's Condition", 1, 10, 7, help="Overall course quality today")
        time_of_day = st.selectbox("Time of Day", options=["Morning", "Afternoon", "Evening"], index=0)

    partner_names_raw = st.text_area(
        "Playing Partners (first name + last initial each, separated by commas/lines)",
        placeholder="e.g., John D, Maria P, Lee K"
    )
    construction = st.checkbox("Construction areas on course?")
    standing_water = st.checkbox("Temporary standing water on course?")

    if st.button("🎯 PREDICT MY SCORE", type="primary"):
        practice_map = {"No": 0, "Driving Range": 1, "Putting Green": 1, "Both": 2}
        greens_map = {"Slow": 0, "Medium": 1, "Fast": 2}
        rough_map = {"Thin": 0, "Normal": 1, "Thick": 2}
        time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2}

        partner_names = parse_partner_names(partner_names_raw)
        num_partners = len(partner_names)

        inputs = {
            'tee_box': tee_box,
            'practice_score': practice_map[practiced],
            'greens_numeric': greens_map[greens_speed],
            'rough_numeric': rough_map[rough_thickness],
            'has_construction': int(construction),
            'has_standing_water': int(standing_water),
            'time_numeric': time_map[time_of_day],
            'num_partners': int(num_partners),
            'physical_condition': int(physical_condition),
            'course_condition': int(course_condition),
            'partner_names': partner_names,
        }

        prediction, features, weather = make_prediction(inputs)
        if prediction is not None:
            st.session_state['last_prediction'] = {
                'score': prediction,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'inputs': inputs,
                'features': features,
                'weather': weather
            }

            st.markdown(f"""
            <div class="prediction-box">
                <h2>🏌️ Predicted Score</h2>
                <div class="prediction-score">{prediction:.0f}</div>
                <p style="font-size: 1.1rem; color: #555;">Expected range: {prediction-3:.0f} - {prediction+3:.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Today's Conditions")
            st.write(f"🌡️ Temperature: {weather['temp_mean']:.0f}°F")
            st.write(f"💨 Wind: {weather['wind_speed_max']:.0f} mph")
            st.write(f"🏌️ Tee Box: {tee_box}")
            st.write(f"💪 Your Physical Condition: {physical_condition}/10")
            st.write(f"🟢 Course's Condition: {course_condition}/10")
            if partner_names:
                st.write("👥 Partners:", ", ".join(partner_names))

            st.success("✅ Prediction saved! Enter your actual score after the round in the next tab.")

# TAB 2: Enter Actual Score
with tab2:
    st.markdown("#### After Your Round")
    if 'last_prediction' not in st.session_state:
        st.info("👈 Make a prediction first in the 'Predict Score' tab!")
    else:
        pred = st.session_state['last_prediction']
        st.markdown(f"**Predicted Score:** {pred['score']:.0f}")
        st.markdown(f"**Date:** {pred['date']}")

        actual_score = st.number_input("Your Actual Score", 70, 130, int(round(pred['score'])))

        if st.button("💾 SAVE SCORE", type="primary"):
            try:
                new_rounds = pd.read_csv(NEW_ROUNDS_FILE)
            except Exception:
                new_rounds = pd.DataFrame(columns=['date', 'predicted_score', 'actual_score', 'partner_names'])

            partner_names = pred['inputs'].get('partner_names', [])
            partner_names_str = ", ".join(partner_names) if partner_names else ""

            new_round = {
                'date': pred['date'],
                'predicted_score': pred['score'],
                'actual_score': actual_score,
                'partner_names': partner_names_str,
                **pred['features']
            }

            new_rounds = pd.concat([new_rounds, pd.DataFrame([new_round])], ignore_index=True)
            new_rounds.to_csv(NEW_ROUNDS_FILE, index=False)

            error = actual_score - pred['score']

            st.markdown(f"""
            <div class="prediction-box">
                <h2>✅ Score Saved!</h2>
                <table style="margin: 20px auto; font-size: 1.2rem;">
                    <tr><td><b>Predicted:</b></td><td>{pred['score']:.0f}</td></tr>
                    <tr><td><b>Actual:</b></td><td>{int(actual_score)}</td></tr>
                    <tr><td><b>Difference:</b></td>
                        <td style="color: {'green' if error < 0 else 'red'};">
                            {'+' if error > 0 else ''}{error:.0f} strokes
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            num_rounds = len(new_rounds.dropna(subset=['actual_score']))
            if num_rounds >= 10 and num_rounds % 10 == 0:
                with st.spinner("🔄 Auto-retraining model..."):
                    df_hist = pd.read_csv(HISTORY_FILE)
                    new_rounds['date'] = pd.to_datetime(new_rounds['date'])
                    new_rounds['score'] = new_rounds['actual_score']
                    combined = pd.concat([df_hist, new_rounds], ignore_index=True)
                    # Make sure combined has all FEATURES
                    for f in FEATURES:
                        if f not in combined.columns:
                            combined[f] = 0
                    train_model(combined)
                    st.success(f"🎉 Model retrained with {num_rounds} new rounds!")
                    st.balloons()
            else:
                next_retrain = ((num_rounds // 10) + 1) * 10
                st.info(f"⏳ {next_retrain - num_rounds} more rounds until next auto-retrain!")

            del st.session_state['last_prediction']

# TAB 3: Stats
with tab3:
    st.markdown("#### 📊 Your Statistics")
    try:
        df_hist = pd.read_csv(HISTORY_FILE)
        st.metric("Total Rounds", len(df_hist))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Score", f"{df_hist['score'].mean():.1f}")
        with col2:
            st.metric("Best Score", int(df_hist['score'].min()))
        with col3:
            st.metric("Worst Score", int(df_hist['score'].max()))

        try:
            new_rounds = pd.read_csv(NEW_ROUNDS_FILE)
        except Exception:
            new_rounds = pd.DataFrame(columns=['date', 'predicted_score', 'actual_score', 'partner_names'])

        if 'actual_score' in new_rounds.columns and len(new_rounds.dropna(subset=['actual_score'])) > 0:
            st.markdown("#### 🆕 New Rounds with Predictions")
            valid_new = new_rounds.dropna(subset=['actual_score'])
            st.metric("Rounds with Predictions", len(valid_new))

            errors = valid_new['actual_score'] - valid_new['predicted_score']
            mae = np.abs(errors).mean()

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Prediction Accuracy (MAE)", f"{mae:.1f} strokes")
            with c2:
                better = (errors < 0).sum()
                st.metric("Beat Prediction", f"{better}/{len(valid_new)}")

            st.markdown("#### 🕐 Last 5 Rounds")
            recent = df_hist.tail(5)[['date', 'score']].copy()
            recent['date'] = pd.to_datetime(recent['date']).dt.strftime('%m/%d/%Y')
            st.dataframe(recent, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading stats: {e}")
