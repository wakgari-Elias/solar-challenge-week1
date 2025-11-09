# Solar Challenge Week 1

## Repository
[https://github.com/wakgari-Elias/solar-challenge-week1](https://github.com/wakgari-Elias/solar-challenge-week1.git)

---

## Objective
Set up a reproducible Python development environment and Git workflow for Solar Challenge Week 1.

---

## Step-by-Step Setup

### 1️⃣ Clone the Repository
```powershell
git clone https://github.com/wakgari-Elias/solar-challenge-week1.git
cd solar-challenge-week1
git checkout -b setup-task

python -m venv venv
.\venv\Scripts\Activate.ps1
git add .
git commit -m "init: add project structure and initial files"
git commit -m "chore: venv setup"
git commit -m "ci: add GitHub Actions workflow"

python -m pip install --upgrade pip
pip install -r requirements.txt
