# ⚡ QUICK START GUIDE

**Get this app live in 15 minutes!**

---

## ✅ Checklist

Before you start, you need:
- [ ] OpenAI API key (get from https://platform.openai.com/api-keys)
- [ ] GitHub account (create at https://github.com)
- [ ] These files downloaded to your computer

---

## 🚀 3-Step Deployment

### 1️⃣ Upload to GitHub (5 min)

1. Go to https://github.com → Click "+" → "New repository"
2. Name: `golf-predictor-app`
3. ✅ Make it **Private**
4. ✅ Add a README
5. Click "Create repository"
6. Click "Add file" → "Upload files"
7. Drag ALL these files into the browser
8. Click "Commit changes"

### 2️⃣ Deploy on Streamlit (5 min)

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `golf-predictor-app`
5. Main file: `app.py`
6. Click "Advanced settings"
7. In "Secrets", paste:
   ```
   OPENAI_API_KEY = "your-key-here"
   ```
   (Replace with your actual OpenAI key!)
8. Click "Deploy"
9. Wait 2-3 minutes...

### 3️⃣ Share with Dad (2 min)

1. Copy the URL (like `https://your-app.streamlit.app`)
2. Text it to your dad
3. Tell him to:
   - Open it on his phone
   - Tap share icon → "Add to Home Screen"
   - Done!

---

## ✨ That's It!

Your dad now has a working golf score predictor on his phone.

**Cost:** ~$1-2 per month in OpenAI API usage  
**Hosting:** FREE on Streamlit Cloud  
**Time investment:** 15 minutes setup, then 0 minutes maintenance  

---

## 📱 What It Looks Like

**On Dad's Phone:**
```
⛳ Golf Score Predictor

[Dropdown menus for conditions]
[Sliders for physical state]

[Big Green Button: PREDICT MY SCORE]

→ Shows: "Predicted Score: 97"

[After Round: Enter actual score]
[Save Score Button]
```

Simple. Clean. Mobile-friendly. Perfect for Dad! 👍

---

## 🆘 If Something Goes Wrong

1. **App won't deploy?**
   - Check that you uploaded ALL files
   - Check that OpenAI key is in secrets
   - Look at the error log in Streamlit dashboard

2. **"API key not configured" error?**
   - Go to Streamlit dashboard → Your app → Settings → Secrets
   - Make sure the key is correct (no extra spaces!)

3. **App is really slow?**
   - First load after inactivity takes 30 sec (Streamlit sleeps)
   - After that, it's fast

4. **Need help?**
   - Read the full README.md
   - Google the error message
   - Check Streamlit docs: https://docs.streamlit.io

---

## 🎉 You Did It!

Now go impress some employers with your other projects! 😉

This one stays private - just a helpful tool for your dad.
