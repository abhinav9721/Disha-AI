# Disha AI – AI Career Path Guide for Rural Youth

Disha AI is an **AI-powered career guidance platform** designed to help rural and underserved youth explore suitable career paths based on their education, skills, interests, goals, available resources, and willingness to move.

The project combines **Python, Flask, Scikit-learn, Machine Learning, SQLite, and a rule-based expert system** to generate personalized career recommendations.

> **Note:** The current ML model is trained on synthetic data generated using domain-based rules. Therefore, the predictions demonstrate the ML pipeline and recommendation system rather than representing statistically validated real-world career outcomes.

---
🚀 Live Demo

🌐 Live Demo: [Available soon]

📂 GitHub Repository: [https://github.com/abhinav9721/Disha-AI]


## 🚀 Key Features

* 🎯 Personalized career recommendations
* 🤖 Machine Learning-based career suitability scoring
* 📊 57 career options
* 🧠 65 numerical input features
* 🌐 Flask-based web application
* 🔐 User login and signup
* 🗄️ SQLite database for user accounts
* 🌍 English and Hindi language support
* 🔎 Career exploration and search
* 📚 Career roadmaps and required skills
* 💰 Approximate training-cost information
* 🏛️ Information about relevant government schemes
* 📱 Rural-youth focused career guidance
* 🖨️ Browser-based print support for results

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Web Framework

* Flask
* Jinja2

### Machine Learning

* Scikit-learn
* Ridge Regression
* Random Forest
* MLP Neural Network
* Ensemble Model

### Database

* SQLite

### Security

* Werkzeug password hashing
* CSRF protection
* Environment variables for sensitive configuration

### Frontend

* HTML
* CSS
* Jinja2 Templates

---

# 📂 Project Structure

```text
disha_ai/
│
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── careers.py
│   ├── labels.py
│   └── i18n.py
│
├── ml/
│   ├── features.py
│   ├── expert.py
│   ├── generate_dataset.py
│   ├── train_model.py
│   └── model_utils.py
│
├── templates/
│   └── *.html
│
├── static/
│   └── style.css
│
└── instance/
    └── disha.db
```

> The database file and other generated/private files should **not be committed to GitHub**.

---

# 🧠 How Disha AI Works

The system follows a simple pipeline:

```text
User Profile
     ↓
Education + Skills + Interests
     ↓
Goals + Internet Access + Mobility
     ↓
Training Budget + Area Type
     ↓
65 Numerical Features
     ↓
Machine Learning Model
     ↓
57 Career Suitability Scores
     ↓
Eligibility Filtering
     ↓
Personalized Career Recommendations
```

---

# 📊 Input Features

The system collects information such as:

* Education level
* Educational stream
* Skills
* Interests
* Main career goal
* Internet availability
* Willingness to relocate
* Training budget
* Area type
* State
* Age
* Other profile information

The ML pipeline converts the relevant profile information into **65 numerical features**.

### Privacy Note

Some profile information may be collected by the application for recommendation or eligibility purposes. Not every collected field is necessarily used as an ML feature.

For example:

* Age, state and gender are not used as ML features.
* Gender is used only for eligibility filtering of specific women-only career roles.

---

# 🤖 Machine Learning Approach

Disha AI treats career recommendation as a **multi-output regression problem**.

For each user profile, the model predicts a suitability score from **0–100 for each of the 57 careers**.

The system evaluates multiple models:

1. Ridge Regression
2. Random Forest
3. MLP Neural Network
4. Random Forest + MLP Ensemble
5. Popularity Baseline

The model with the lowest Mean Absolute Error (MAE) on the hold-out test set is selected.

---

# 📈 Example Model Results

One training run produced approximately the following results:

| Model               |  MAE |   R² | Hit@1 | Precision@3 |
| ------------------- | ---: | ---: | ----: | ----------: |
| Popularity Baseline | 13.1 | 0.00 |  0.10 |        0.10 |
| Ridge Regression    |  3.6 | 0.92 |  0.85 |        0.68 |
| Random Forest       |  6.9 | 0.69 |  0.74 |        0.57 |
| MLP Neural Network  |  3.5 | 0.93 |  0.84 |        0.67 |
| RF + MLP Ensemble   |  4.6 | 0.87 |  0.81 |        0.64 |

**Important:** These numbers can vary between training runs depending on the generated data and model configuration.


# ⚠️ Important: Synthetic Training Data

The current version of Disha AI uses **synthetically generated training data**.

There is no single public dataset that directly maps:

```text
Education + Skills + Interests + Goals
                ↓
        Suitable Career
```

for Indian rural youth at the required level of detail.

Therefore, the project generates realistic user profiles and assigns career suitability scores using an **expert/domain-based scoring rule**, with controlled random noise.

The ML model then learns these generated patterns.

### What this means

The current model:

* Demonstrates an end-to-end ML recommendation pipeline.
* Learns patterns generated by the expert scoring system.
* Does not independently discover new truths about career suitability.
* Should not be interpreted as a scientifically validated career prediction model.

This distinction is important when presenting the project in a portfolio, interview, or academic evaluation.

---

# 🔄 Making the Model Data-Driven

A future version can be trained using real-world data collected through:

* Youth surveys
* ITI institutions
* KVKs
* College placement cells
* Skill-development programs
* Career counselling organizations
* Publicly available datasets

The real dataset should contain equivalent feature columns and career suitability targets.

Example:

```bash
python -m ml.train_model --data my_real_data.csv
```

Using real data can help evaluate whether the learned relationships generalize beyond the synthetic expert rules.

---

# 🎯 Career Recommendation Process

After prediction, Disha AI performs an additional eligibility check.

If a career requires an education level higher than the user's current education, it can be placed under:

```text
After More Study
```

The recommendation system also generates explanations such as:

* Why the career may fit the profile
* Skills that should be developed
* Education requirements
* Suggested learning roadmap
* Approximate training cost
* Relevant schemes

The explanation is generated from the project's domain-knowledge rules.

---

# 🌐 Application Pages

The Flask application includes:

### Login / Signup

Users can create an account and securely authenticate.

### Career Assessment

A multi-step form collects the information required for recommendations.

### Recommendations

The system displays suitable career options with match scores and supporting information.

### Explore

Users can search and explore available career paths.

### Career Details

Each career can contain:

* Required skills
* Education
* Learning roadmap
* Approximate cost
* Relevant schemes
* Other career information



## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

in your browser.

---

# 🧪 Train the ML Model

The first application startup can train/load the required model depending on the project configuration.

To manually generate/train the model:

```bash
python -m ml.train_model
```

Generated model files should be handled according to the project's `.gitignore` policy if they are not intended to be stored in the repository.

# 📌 Current Limitations

Disha AI is currently a project/prototype and has several limitations:

* Training data is synthetic.
* Career suitability scores are based on domain rules.
* Income information is approximate.
* Eligibility requirements may change over time.
* Government schemes and training information may change.
* Model performance on synthetic data does not guarantee real-world performance.
* The system should be treated as a career exploration and guidance tool, not as a definitive career decision-maker.

---

# 🔮 Future Improvements

Possible future improvements include:

* Real-world training dataset
* Cross-validation
* Hyperparameter optimization
* SHAP-based explanations
* Feature importance visualization
* Model information dashboard
* State-wise job information
* State-wise government schemes
* Live job-market integration
* Skill-course recommendations
* Multilingual voice assistance
* Mobile application
* Better accessibility for rural users
* Periodic updating of career and scheme information

---

# 🎓 Project Objective

The main objective of Disha AI is to demonstrate how **Artificial Intelligence and Machine Learning can be used to build accessible career-guidance tools for rural and underserved communities**.

The project focuses on connecting a user's:

```text
Education
+
Skills
+
Interests
+
Goals
+
Available Resources
        ↓
Career Exploration
        ↓
Personalized Recommendations
        ↓
Learning Roadmap
```

---

# 👨‍💻 Project Type

**Domain:** Artificial Intelligence / Machine Learning / Career Guidance

**Application:** Web Application

**Framework:** Flask

**ML Library:** Scikit-learn

**Database:** SQLite

**Language:** Python

---

# 📄 License

This project is intended for educational and portfolio purposes.

If you plan to distribute or reuse the project, add an appropriate open-source license such as MIT License after reviewing its terms.

---

# ⭐ Acknowledgement

This project was developed as an academic/portfolio project to explore the practical use of **Machine Learning, Flask, recommendation systems, and rule-based expert knowledge** in career guidance.

If you find the project useful, consider giving the repository a ⭐ on GitHub.
 
 
 👨‍💻 Created By
Abhinav Tripathi

B.Tech CSE (AI) Student

Passionate about Artificial Intelligence, Machine Learning, Data Science, and Python Development.

🔗 LinkedIn: Abhinav Tripathi
