#!/bin/bash
# ============================================================
# ALIVE Frontend Deploy Script
# Repo: https://github.com/Medi-Edu/Medical-Imaging
# File: app.js  (root of main branch)
# ============================================================

set -e   # stop on any error

# ── STEP 1: Clone the repo (skip if already cloned) ─────────────────────────
if [ ! -d "Medical-Imaging" ]; then
  echo "Cloning Medi-Edu/Medical-Imaging..."
  git clone https://github.com/Medi-Edu/Medical-Imaging.git
else
  echo "Repo already cloned — pulling latest..."
  git -C Medical-Imaging pull origin main
fi

# ── STEP 2: Copy the updated app.js into the repo ───────────────────────────
echo "Copying updated app.js..."
cp app.js Medical-Imaging/app.js

# ── STEP 3: Commit and push ──────────────────────────────────────────────────
cd Medical-Imaging
git add app.js
git commit -m "fix: hallucination dedup, /ask_avatar endpoint, UUID temp files, Cloudflare localStorage fallback"
git push origin main

echo ""
echo "Done. GitHub Pages will rebuild in ~30-60 seconds."
echo "Check: https://medi-edu.github.io/Medical-Imaging/"
