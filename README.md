# ⛳ Golf Score Predictor App

A mobile-friendly web app that predicts golf scores using machine learning and provides personalized insights.

**Built for Dad with ❤️**

---

## 📱 What This App Does

- Predicts your golf score before each round
- Learns from your actual scores over time
- Auto-retrains the model every 10 rounds
- Works perfectly on mobile devices
- Simple, clean interface

---

## 🚀 Deployment Instructions

Follow these steps to get the app running online for free!

### Step 1: Get Your OpenAI API Key (5 minutes)

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Give it a name (e.g., "Golf App")
5. **COPY THE KEY** (you won't see it again!)
6. Keep it somewhere safe temporarily

**Cost:** ~$0.10 per round (you'll pay for this)

---

### Step 2: Create GitHub Account (If You Don't Have One)

If you already have GitHub, skip to Step 3.

1. Go to https://github.com
2. Click "Sign up"
3. Follow the prompts
4. Verify your email

---

### Step 3: Upload Code to GitHub (10 minutes)

#### Option A: Using GitHub Website (Easiest)

1. Go to https://github.com
2. Click the **"+"** in top right → **"New repository"**
3. Repository name: `golf-predictor-app`
4. Description: "Golf score predictor for Dad"
5. ✅ Check **"Private"** (important!)
6. ✅ Check **"Add a README file"**
7. Click **"Create repository"**

8. Now upload files:
   - Click **"Add file"** → **"Upload files"**
   - Drag and drop ALL files from this folder:
     - `app.py`
     - `requirements.txt`
     - `.gitignore`
     - `.streamlit/` folder (with `secrets.toml.example` inside)
   - Click **"Commit changes"**

#### Option B: Using Git Command Line (If You Know Git)

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/golf-predictor-app.git
git push -u origin main
```

---

### Step 4: Create Streamlit Cloud Account (2 minutes)

1. Go to https://streamlit.io/cloud
2. Click **"Sign up"**
3. Choose **"Continue with GitHub"**
4. Authorize Streamlit to access your GitHub

---

### Step 5: Deploy Your App (5 minutes)

1. In Streamlit Cloud, click **"New app"**

2. Fill in:
   - **Repository:** `your-username/golf-predictor-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Choose a custom name (e.g., `dads-golf-app`)

3. Click **"Advanced settings"**

4. Under **"Secrets"**, paste this (with YOUR actual OpenAI key):
   ```toml
   OPENAI_API_KEY = "sk-your-actual-openai-key-here"
   ```
   ⚠️ Replace `sk-your-actual-openai-key-here` with your real key!

5. Click **"Deploy!"**

6. Wait 2-3 minutes for it to deploy...

7. **Done!** Your app is live! 🎉

---

### Step 6: Get Your App URL

After deployment completes, you'll see:

```
https://your-app-name.streamlit.app
```

**This is the link your dad will use!**

---

## 📱 How Your Dad Uses It

### Setup (One Time):

1. Send him the app URL
2. Have him open it on his phone
3. Tell him to tap the **share icon** → **"Add to Home Screen"**
4. Now it appears like a real app on his home screen!

### Before Each Round:

1. Open the app
2. Go to "🎯 Predict Score" tab
3. Fill out the form (takes 30 seconds)
4. Tap **"PREDICT MY SCORE"**
5. See the prediction!

### After Each Round:

1. Go to "📝 Enter Actual Score" tab
2. Enter his score
3. Tap **"SAVE SCORE"**
4. Done! The app learns from this data

### View Stats:

1. Go to "📊 Stats" tab
2. See all statistics and history

---

## 🔐 Security Notes

✅ **Repository is PRIVATE** - Only you can see the code  
✅ **API key is in Streamlit secrets** - Not visible in code  
✅ **HTTPS encrypted** - All data is secure  
✅ **No personal data** - Just golf scores  

---

## 💰 Cost Breakdown

- **Streamlit Cloud:** FREE forever
- **GitHub:** FREE for private repos
- **OpenAI API:** ~$0.10 per round

**Monthly cost if dad plays 12 rounds:** ~$1.20/month

---

## 🛠️ Troubleshooting

### App won't deploy?
- Check that `requirements.txt` was uploaded
- Check that you added the OpenAI API key to secrets
- Check the logs in Streamlit Cloud for errors

### "OpenAI API key not configured"?
- Go to Streamlit Cloud dashboard
- Click your app → ⚙️ Settings
- Go to "Secrets"
- Make sure the key is there and correct

### App is slow?
- Streamlit free tier can sleep after inactivity
- First load after sleep takes ~30 seconds
- After that, it's fast

### Dad can't find the app?
- Make sure he bookmarked it or added to home screen
- Send him the full URL via text

---

## 🔄 Updating the App

If you want to make changes:

1. Edit `app.py` on your computer
2. Go to GitHub repository
3. Click `app.py` → Edit (pencil icon)
4. Paste your changes
5. Click "Commit changes"
6. Streamlit will automatically redeploy! (takes 2 min)

---

## 📊 Features

- ✅ Mobile-optimized design
- ✅ Predicts scores using 25+ features
- ✅ Auto-retrains every 10 rounds
- ✅ Tracks prediction accuracy
- ✅ Shows statistics and trends
- ✅ Works offline after first load
- ✅ No app store needed

---

## 🏗️ Technical Details (For You)

- **Framework:** Streamlit
- **ML Model:** XGBoost Regressor
- **API:** OpenAI GPT-4 (for future feedback feature)
- **Storage:** CSV files (persisted in Streamlit Cloud)
- **Hosting:** Streamlit Cloud (free tier)

### Model Features:
- Course conditions (tee box, greens, rough)
- Weather (temperature, wind, precipitation)
- Personal factors (practice, physical condition, playing partners)
- Historical performance (rolling averages, trends)
- Temporal features (day of week, month, days since last round)

---

## 📝 To-Do (Future Enhancements)

- [ ] Add AI-generated feedback (already integrated, just needs UI)
- [ ] Export data to CSV
- [ ] Compare rounds side-by-side
- [ ] Share predictions with friends
- [ ] Dark mode
- [ ] Multiple courses support

---

## 🙏 Support

If something breaks or you need help:

1. Check Streamlit Cloud logs (in dashboard)
2. Check GitHub issues (if you made repo public)
3. Google the error message
4. Message the Streamlit community: https://discuss.streamlit.io/

---

## ❤️ Credits

Built by a loving son for his dad.

Uses:
- Streamlit (for the web app)
- XGBoost (for predictions)
- OpenAI (for future AI feedback)
- 50 rounds of historical GHIN data

---

**Enjoy the app, Dad! ⛳**
