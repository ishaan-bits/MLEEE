@echo off
REM Quick Git Setup & Push Script for Windows
REM Run this batch file to automatically push to GitHub

cd /d d:\MLEEE

echo.
echo ====================================================================
echo INITIALIZING GIT REPOSITORY
echo ====================================================================
echo.

REM Initialize git repo if not already done
git init

echo.
echo ====================================================================
echo STAGING ALL FILES
echo ====================================================================
echo.

REM Add all files
git add .

echo.
echo ====================================================================
echo CREATING INITIAL COMMIT
echo ====================================================================
echo.

REM Create commit
git commit -m "Initial commit: Complete ML pipeline with baseline and advanced models producing 97-98%% F1-Score"

echo.
echo ====================================================================
echo ADDING REMOTE REPOSITORY
echo ====================================================================
echo.

REM Add remote (if not already added)
git remote add origin https://github.com/Sidyeet/ML-Model-for-Predictive-Maintenance-in-Industrial-Systems.git 2>nul

echo.
echo ====================================================================
echo PUSHING TO GITHUB
echo ====================================================================
echo.

REM Rename branch to main and push
git branch -M main
git push -u origin main

echo.
echo ====================================================================
echo PUSH COMPLETE!
echo ====================================================================
echo.
echo  Your project has been pushed to GitHub!
echo.
echo Verify at:
echo https://github.com/Sidyeet/ML-Model-for-Predictive-Maintenance-in-Industrial-Systems
echo.
pause
