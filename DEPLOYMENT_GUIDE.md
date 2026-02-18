# Deployment Guide

## How to Deploy to Render.com and Get Your URL

### Step 1: Go to Render Dashboard
1. Visit https://dashboard.render.com
2. Sign up/Login with GitHub

### Step 2: Create New Web Service
1. Click **"New +"** → **"Web Service"**
2. Select your GitHub repository: `flask-tts-api2`
3. Configure:
   - **Name**: `tts-api`
   - **Environment**: `Docker`
   - **Branch**: `main`

### Step 3: Add Environment Variables
In the dashboard, add these secrets (click "Advanced" → "Add Environment Variable"):

| Key | Value |
|-----|-------|
| `TWILIO_ACCOUNT_SID` | Your Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Your Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Your Twilio phone number (e.g., +1234567890) |
| `BASE_URL` | (will be your app URL after deploy) |

### Step 4: Deploy
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for Docker build
3. You'll get a URL like: `https://tts-api.onrender.com`

### Step 5: Update BASE_URL
After deployment, copy your Render URL and add it as environment variable:
- Key: `BASE_URL`
- Value: `https://tts-api.onrender.com` (your actual URL)

---

## Test Your API

**Health Check:**
```bash
curl https://your-app.onrender.com/health
```

**Generate Speech:**
```bash
curl -X POST https://your-app.onrender.com/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  -o output.wav
```

**Generate Speech + Phone Call:**
```bash
curl -X POST https://your-app.onrender.com/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is your pharmacy assistant calling.", "phone_number": "+919999999999"}'
```

---

## Need Twilio Account?
1. Go to https://www.twilio.com
2. Sign up for free
3. Get Account SID & Auth Token from console
4. Buy a phone number ($1-2/month)

