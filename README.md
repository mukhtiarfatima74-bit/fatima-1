# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an **ATS score (0-100)**, a category breakdown,
keyword gaps and concrete, prioritised improvements - powered by **Google Gemini Flash** and built with **Streamlit**.

## Features
- Reads PDF, DOCX (including tables) and TXT resumes
- Optional job description + target role for a much more accurate keyword match
- Overall ATS score computed from 5 weighted categories:
  keyword match (30%), content & impact (25%), ATS-friendly formatting (20%), structure (15%), readability (10%)
- Prioritised improvements (High / Medium / Low), before/after rewrite examples, missing vs. found keywords
- Shows the extracted text, i.e. what an ATS actually "sees"
- Download the full report as JSON
- Friendly errors for scanned PDFs, wrong API key, rate limits; automatic model fallback

## Project structure
```
.
├── app.py              # the whole app
├── requirements.txt    # dependencies
├── README.md
└── .gitignore          # keeps secrets out of GitHub
```

## Run locally
1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install and run:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Provide your key in **one** of these ways:
   - paste it into the sidebar when the app opens, or
   - create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     # GEMINI_MODEL = "gemini-3.5-flash"   # optional override
     ```
   - or set an environment variable: `export GEMINI_API_KEY=your-key-here`

## Choosing the model
The default is `gemini-3.5-flash`. Google renames models regularly, so you can change it in the
sidebar or via the `GEMINI_MODEL` secret. If the model isn't found, the app automatically tries
`gemini-2.5-flash` and `gemini-flash-latest`. Edit `DEFAULT_MODELS` at the top of `app.py` to change the list.

## Push to GitHub (using the website, no Git needed)
1. Sign in at https://github.com and click **+ → New repository**.
2. Name it (e.g. `ats-resume-checker`), choose **Public**, and click **Create repository**.
3. On the empty repo page click **uploading an existing file**.
4. Drag in `app.py`, `requirements.txt`, `README.md` and `.gitignore`.
   Upload the *files*, not the parent folder, so `app.py` sits at the repo root.
5. Click **Commit changes**.

> ⚠️ Never upload `secrets.toml` or paste your API key into any file in the repo.

## Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub (authorize access if asked).
2. Click **Create app** (or **New app**) → deploy from an existing repo.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Click **Advanced settings**, pick a recent Python version (3.11 or 3.12), and in **Secrets** paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. In a couple of minutes you'll get a public `https://<name>.streamlit.app` URL.

To update the live app, edit the file on GitHub (pencil icon → Commit) - Streamlit redeploys automatically.
To change secrets later: app dashboard → **⋮ → Settings → Secrets**.

> If you put your key in Secrets, anyone with the public link can use your quota. Leave the secret out
> to make visitors enter their own key in the sidebar, or set a spending cap in Google AI Studio.

## How the score works
Gemini rates five categories from 0-100 and returns structured JSON. The overall score is then
calculated in Python from the weights above, so it's transparent and consistent. It is an
**estimate**: real ATS products (Workday, Greenhouse, Lever, Taleo...) don't publish their rules,
so use it as a guide to improve the resume, not as a guarantee.

## Privacy
The resume text is sent to the Gemini API for analysis. The app itself stores nothing on disk.

## Tips for a higher score
- Use a single-column layout with standard headings (Summary, Experience, Education, Skills)
- Avoid text inside images, headers/footers and complex tables
- Mirror keywords from the job description where they are truthful
- Start bullets with action verbs and include measurable results
