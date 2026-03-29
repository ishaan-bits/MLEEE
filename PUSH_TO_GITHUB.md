# 📤 Push to GitHub Instructions

## Automated Method (Recommended)

If you have GitHub Desktop or VS Code with git integration:

### Option 1: GitHub Desktop
1. Open GitHub Desktop
2. Click "Add Local Repository"
3. Browse to `d:\MLEEE` and select it
4. Click "Create a Repository"
5. Select "Local" as the local path
6. Enter repository details (already filled if recognized)
7. Click "Publish repository"  "Publish"

### Option 2: VS Code Git Integration
1. Open `d:\MLEEE` in VS Code
2. Open Terminal (Ctrl + `)
3. Run: `git init`
4. Stage files: `git add .`
5. Commit: `git commit -m "Initial commit: ML pipeline with baseline and advanced models"`
6. Add remote: `git remote add origin https://github.com/Sidyeet/ML-Model-for-Predictive-Maintenance-in-Industrial-Systems.git`
7. Push: `git branch -M main && git push -u origin main`

## Command Line Method

If you have Git Bash installed:

```bash
cd d:/MLEEE

# Initialize repository (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Complete ML pipeline with preprocessing, two trained models, and evaluation metrics"

# Add remote repository
git remote add origin https://github.com/Sidyeet/ML-Model-for-Predictive-Maintenance-in-Industrial-Systems.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## What Gets Pushed

 **Code Files:**
- `src/data_loader.py`
- `src/preprocessing.py`
- `src/feature_selection.py`
- `src/models.py`
- `src/evaluation.py`

 **Notebooks:**
- `notebooks/01_EDA.ipynb`
- `notebooks/02_Presentation.ipynb`

 **Documentation:**
- `README.md`
- `requirements.txt`
- `.gitignore`

 **Results:**
- `results/metrics.json`
- Saved models: `models/baseline_lr.pkl`, `models/advanced_rf.pkl`
- Processed data: `data/processed/train.csv`, `data/processed/test.csv`

## Notes

- The `.gitignore` file prevents large binary files and __pycache__ from being uploaded
- The repository should already exist at the GitHub link provided
- Make sure you're authenticated with GitHub (VS Code may prompt for login)
- If you get authentication errors, generate a Personal Access Token from GitHub settings

## Verify Push Success

After pushing, verify on GitHub:
```
https://github.com/Sidyeet/ML-Model-for-Predictive-Maintenance-in-Industrial-Systems
```

You should see:
- All Python files in `src/`
- Notebooks in `notebooks/`
- README and requirements.txt
- results/ folder with metrics.json
- All commits in the history

## Future Commits

After the initial push, future updates are simple:

```bash
git add .
git commit -m "Descriptive message about changes"
git push
```
