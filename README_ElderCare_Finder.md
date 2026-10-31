# ElderCare Finder
### Adaptive Old Age Home Recommendation System

> A user-centered, full-stack recommendation platform for discovering and comparing old-age homes across India.

ElderCare Finder combines **content-based Recommendation Systems (RS)** with an experimental **contextual-bandit Reinforcement Learning (RL)** layer. Users can create an account, complete a short personalization questionnaire, search and filter listings, inspect recommendation explanations, and submit feedback that may influence future rankings.

The project uses `data/All_India_Old_Age_Homes.xlsx`, worksheet `Master_Data`, with **518 source records**.

---

## Table of Contents

- [Highlights](#highlights)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Dataset and Data Quality](#dataset-and-data-quality)
- [Recommendation System (RS)](#recommendation-system-rs)
- [Reinforcement Learning (RL)](#reinforcement-learning-rl)
- [Installation and Local Setup](#installation-and-local-setup)
- [Configuration](#configuration)
- [API Overview](#api-overview)
- [Run the RL Simulation](#run-the-rl-simulation)
- [Evaluation Plan](#evaluation-plan)
- [Project Structure](#project-structure)
- [Limitations and Responsible Use](#limitations-and-responsible-use)
- [References](#references)
- [Author](#author)

---

## Highlights

- Account registration and login using Argon2 password hashing and JWT sessions.
- Guided onboarding questionnaire with editable preferences.
- Responsive React interface with search, filters, recommendation cards, explanations, data-quality warnings, contact links and ratings.
- Content-based recommendations using TF-IDF and cosine similarity.
- Hard filters for state, district, facility category, care type, gender eligibility and management type.
- Experimental per-user contextual bandit that updates from explicit 1–5 ratings.
- Frontend control to compare baseline and adaptive ranking.
- Feedback summary metrics.
- SQLite by default; PostgreSQL supported through `DATABASE_URL`.
- Missing or zero capacity and absent contact details are surfaced rather than fabricated.

## How It Works

1. **Sign in or register.** The backend authenticates the user and issues a JWT.
2. **Personalize.** The user answers a short questionnaire about location, care needs and preferences.
3. **Filter candidates.** Mandatory filters remove homes that do not meet selected requirements.
4. **Rank candidates.** TF-IDF and cosine similarity estimate textual relevance; contact completeness contributes a small additional signal.
5. **Adapt (optional).** The contextual-bandit prototype adjusts ranking using learned estimates and an uncertainty bonus.
6. **Give feedback.** A rating is converted into a reward and updates the per-user bandit statistics.

> **Important:** a match score is a ranking signal, not a probability of suitability or a measure of care quality.

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, JavaScript/JSX, Vite, CSS |
| Backend | Python, FastAPI |
| Recommendation | Pandas, scikit-learn, TF-IDF, cosine similarity, NumPy |
| Persistence | SQLAlchemy, SQLite by default, PostgreSQL optional |
| Authentication | Argon2 password hashing, JWT |
| Adaptive ranking | Linear contextual-bandit prototype |
| Dataset | Excel workbook: `All_India_Old_Age_Homes.xlsx` |

## Architecture

```text
React + Vite frontend (port 5173)
 ├── Login / registration
 ├── Personalization questionnaire
 ├── Search and recommendation dashboard
 └── Ratings and preference controls
             │ HTTP requests + JWT
             ▼
FastAPI backend (port 8000)
 ├── Authentication and user profiles
 ├── Dataset import and hard eligibility filters
 ├── TF-IDF + cosine-similarity recommender
 ├── Contextual-bandit ranking and feedback updates
 └── SQLAlchemy persistence
             │
             ├── SQLite (default)
             └── PostgreSQL (optional DATABASE_URL)
```

Bandit parameters are saved separately as JSON files under `backend/artifacts/`.

## Dataset and Data Quality

**Primary dataset:** `data/All_India_Old_Age_Homes.xlsx`  
**Worksheet:** `Master_Data`  
**Source rows inspected:** 518

The workbook includes listing information such as home name, location, address, facility type, care type, gender eligibility, management, reported bed capacity, contact details and source fields.

Data-handling principles:

- Preserve source information and provenance where available.
- Do not invent missing fees, facilities, capacity, vacancies or quality indicators.
- Treat absent or zero capacity as unverified rather than proof that a home has no capacity.
- Treat questionnaire selections as preferences, not evidence that a home provides a service.
- Confirm important details directly with the facility.

The workbook is a directory, not a verified live inventory. Source counts and details may change when the dataset is updated.

## Recommendation System (RS)

The baseline recommender uses a content-based pipeline.

### 1. Candidate generation and hard filters

Selected filters narrow the directory by state, district, facility type, care type, gender eligibility and management type. These constraints are applied before ranking. The adaptive layer should only reorder candidates that survive the mandatory filters.

### 2. TF-IDF text representation

Descriptive listing fields are combined into searchable text and vectorized using scikit-learn's `TfidfVectorizer`. The current configuration uses `ngram_range=(1, 2)`, `max_features=8000`, and `stop_words="english"`.

### 3. Cosine similarity

The query is transformed using the same vectorizer and compared with candidate listings using cosine similarity. Higher similarity indicates stronger lexical alignment; it does **not** guarantee suitability.

### 4. Contact-completeness adjustment

```text
contact_completeness =
    0.50 × has_telephone
  + 0.20 × has_email
  + 0.30 × has_address

baseline_score =
    0.85 × cosine_similarity
  + 0.15 × contact_completeness
```

These weights are manually selected baseline parameters, not empirically optimized values. Contact completeness measures the presence of directory fields—not care quality, safety or service reliability.

### 5. Questionnaire preferences

Questionnaire answers support more relevant searches. Some profile answers are used as search text or fallback filters, while others are stored for future development. Budget is collected but is not treated as a factual fee filter because reliable fee data is unavailable in the supplied directory.

## Reinforcement Learning (RL)

The adaptive component is an **experimental linear contextual bandit**. It learns from individual recommendation feedback, but does not model long sequences of environment transitions as a full Markov Decision Process (MDP) would.

### Context features

The current feature vector includes a bias term, query-to-home text similarity, telephone/email/address availability, whether positive bed capacity is known, and whether a non-empty query was supplied. It does not directly encode every questionnaire answer; explicit user–home context features are a recommended extension.

### Action and reward

The system scores eligible candidates and uses the adaptive score to influence their order. Explicit ratings are converted to rewards:

```text
reward = (rating - 3) / 2
```

| Rating | Reward |
|---:|---:|
| 1 | -1.0 |
| 2 | -0.5 |
| 3 | 0.0 |
| 4 | +0.5 |
| 5 | +1.0 |

### Learning update

The learner maintains per-user linear sufficient statistics, commonly denoted by `A` and `b`:

```text
A ← A + x xᵀ
b ← b + reward × x
theta_hat = pseudoinverse(A) × b
```

The predicted reward is estimated from the candidate feature vector and learned parameters. The live adaptive ranking adds an uncertainty bonus and blends the adaptive signal with the content-based baseline. Parameters are saved in files such as `backend/artifacts/bandit_user_<user_id>.json`.

### Safety boundary and limitations

Mandatory filters are applied before adaptive scoring. The bandit must not reintroduce a home excluded by a mandatory filter.

The live ranker uses an uncertainty bonus; a separate simulation demonstrates epsilon-greedy and UCB explicitly. These are distinct parts of the project. This prototype does not establish that adaptive recommendations outperform the static baseline. Representative feedback or a clearly labelled simulated-user experiment is needed for evaluation.

## Installation and Local Setup

### Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer
- npm

### 1. Start the backend

Open PowerShell in the project root:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On first startup, the application creates database tables and imports records from `../data/All_India_Old_Age_Homes.xlsx`.

- API documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/api/health>

### 2. Start the frontend

Open a **second** PowerShell terminal from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>, register an account and complete the questionnaire. Vite proxies `/api` requests to FastAPI.

### Troubleshooting

- **Workbook not found:** verify the workbook exists at `data/All_India_Old_Age_Homes.xlsx` relative to the project root.
- **Frontend cannot reach the API:** ensure the backend is running on port `8000` and the Vite proxy configuration is correct.
- **Database changes are not reflected:** check `DATABASE_URL`; changing it may point the app to a different database.

## Configuration

### SQLite (default)

No database configuration is required for a basic local run.

### Optional PostgreSQL

Create a PostgreSQL database, install the driver listed in backend requirements, and set `DATABASE_URL` before starting FastAPI:

```powershell
$env:DATABASE_URL="postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/eldercare"
$env:JWT_SECRET="replace-with-a-long-random-secret"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Replace the example password and secret. Use a long, unique secret outside local development. **Never commit passwords, production secrets, tokens or user databases to Git.** The default secret is for local demo use only. Production deployment requires secure secret management and a security review.

## API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check API status and listing count |
| `GET` | `/api/options` | Retrieve filter options |
| `POST` | `/api/auth/register` | Register a user |
| `POST` | `/api/auth/login` | Authenticate and receive a token |
| `GET` | `/api/me` | Retrieve the authenticated user's profile |
| `PUT` | `/api/profile` | Save questionnaire preferences |
| `POST` | `/api/recommendations` | Filter and rank candidate homes |
| `POST` | `/api/feedback` | Record a rating and update adaptive parameters |
| `GET` | `/api/metrics` | Retrieve feedback and prototype summary metrics |

Visit <http://127.0.0.1:8000/docs> while the backend is running for the interactive API documentation and exact request schemas.

## Run the RL Simulation

From the `backend` directory:

```powershell
python experiments/bandit_simulation.py
```

The experiment compares **epsilon-greedy** and **Upper Confidence Bound (UCB)** on a synthetic multi-armed bandit and reports cumulative reward and regret.

This is an algorithm demonstration only. The reward probabilities are synthetic, not learned from the old-age-home directory, and must not be presented as real-world recommender performance.

**Syllabus connection**
- **RS:** content-based filtering, TF-IDF, cosine similarity, candidate filtering and ranking evaluation.
- **RL:** context, action, reward, exploration versus exploitation, epsilon-greedy, UCB, cumulative reward and regret.

A full MDP environment, Bellman-equation-based value functions, policy iteration, value iteration and Monte Carlo control are not part of the current live ranking path.

## Evaluation Plan

Compare static content-based ranking with adaptive ranking using the same queries and eligibility constraints.

## Search-Level Recommendation Evaluation

ElderCare Finder displays ranking evaluation metrics alongside
each recommendation search.

The evaluation is based on explicit user feedback collected
for the same search query and filter combination.

### Metrics

**1. Precision@5**

Measures the proportion of explicitly rated homes in the
top five recommendations that users marked relevant.

Precision@5 = Relevant rated homes in top 5 /
              Total rated homes in top 5

**2. Recall@5**

Measures the proportion of all relevant rated homes in the
search feedback history that appear in the top five.

Recall@5 = Relevant rated homes in top 5 /
           Total relevant rated homes in search history

**3. NDCG@5**

Measures ranking quality while considering the position
and graded relevance of each rated recommendation.

Higher NDCG indicates that more relevant homes appear
towards the top of the ranking.

### Feedback relevance mapping

| Rating | Interpretation | NDCG relevance grade |
|---|---|---:|
| 1 | Not relevant | 0 |
| 2 | Not relevant | 0 |
| 3 | Neutral | 1 |
| 4 | Relevant | 2 |
| 5 | Highly relevant | 3 |

### Evaluation workflow

1. User performs a search.
2. The recommender generates and ranks eligible homes.
3. The evaluation panel displays available metrics.
4. User rates recommended homes from 1 to 5.
5. Feedback is stored with the search identifier.
6. Metrics are recalculated after feedback is submitted.

### Important limitations

- These are feedback-based observed metrics.
- Unrated homes are not automatically considered irrelevant.
- Recall is limited to relevant homes rated in the matching
  search history.
- Ratings are subjective and self-selected.
- These values are not an independent ground-truth benchmark.
- A separate relevance-labelled test dataset is needed for
  rigorous offline evaluation.

### RS metrics

- **Precision@K:** proportion of the top K results judged relevant.
- **Recall@K:** proportion of all judged-relevant candidates retrieved in the top K.
- **NDCG@K:** ranking quality with greater weight on relevant results near the top.
- **Catalog coverage:** breadth of homes recommended across searches.
- **Filter compliance:** whether recommendations respect mandatory constraints.

### RL metrics

- **Cumulative and average reward:** feedback reward over recommendation decisions.
- **Cumulative regret:** reward opportunity lost relative to the experiment's benchmark policy.
- **Exploration behaviour:** how often alternatives are explored.
- **Policy updates:** number of feedback-driven learner updates.

### Suggested protocol

1. Establish the content-based baseline.
2. Create a judged relevance set using appropriate human or expert assessments.
3. Keep held-out queries or feedback for testing.
4. Compare static and adaptive rankings under identical hard filters.
5. If real feedback is insufficient, run a clearly labelled simulated-user experiment.
6. Perform ablations, such as removing contact completeness or the uncertainty bonus.
7. Report dataset size, feedback coverage and limitations.

Do not report training feedback as test performance, and do not claim RL improves quality until a fair comparison supports that conclusion.

## Project Structure

```text
eldercare_adaptive_final/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── db.py
│   │   ├── models.py
│   │   ├── security.py
│   │   └── rl_ranker.py
│   ├── experiments/
│   │   └── bandit_simulation.py
│   ├── requirements.txt
│   └── artifacts/              # runtime-generated learner files
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
├── data/
│   └── All_India_Old_Age_Homes.xlsx
└── README.md
```

Runtime-generated files, local databases, credentials and secrets should generally be excluded from version control.

## Limitations and Responsible Use

- The workbook is a directory, not a verified live inventory.
- Current vacancies, monthly fees, service guarantees, inspection outcomes and care quality are not reliably established by the supplied dataset.
- Missing information means unknown; it does not prove that a service or facility is absent.
- Questionnaire facility selections are preferences, not verified service facts.
- Budget is stored but does not act as a reliable fee filter.
- Ratings are sparse and self-selected; early results are preliminary.
- The adaptive ranker has not been proven superior to the baseline.
- Saved-home state and some user-experience features may be limited to the current frontend session.
- Avoid entering sensitive medical details in search queries. Collect only necessary information and give users control over saved preferences.
- Confirm fees, vacancies, eligibility and care arrangements directly with the facility.
- Match scores are not probabilities, medical advice, quality ratings or guarantees of suitability.

Before production use, add database migrations, stronger operational security, rate limiting, robust audit and feedback logs, privacy/retention controls, automated tests and a review of the recommendation policy.

## References

1. **Elderly Care India** — Old-age-home directory: <https://elderlycareindia.org/list-of-old-age-homes-in-india/>
2. **AIKosh** — Integrated Programme for Older Persons grant data: <https://aikosh.indiaai.gov.in/web/datasets/details/integrated_programme_for_older_persons_goi_grant_data.html>
3. **data.gov.in** — Integrated Programme for Older Persons: <https://www.data.gov.in/resource/integrated-programme-older-persons-goi-grant>
4. **Department of Social Justice and Empowerment, Government of India** — Integrated Programme for Senior Citizens (IPSrC): <https://www.dosje.gov.in/schemes-and-services/integrated-programme-for-senior-citizens-ipsrc/>
5. **NemaCare** — Guide to old-age homes in India: <https://www.nemacare.com/post/old-age-homes-in-india-the-complete-end-to-end-guide>
6. **Elkin** — Directory: <https://www.elkin.co.in/>

These references provide background and directory/programme context. They do not validate the prototype's recommendation accuracy.

## Author

**Kanishka Gole**  
MBA Tech (Artificial Intelligence) · NMIMS Mumbai

---

*ElderCare Finder is an academic prototype intended to demonstrate recommendation-system methods and feedback-driven adaptive ranking. It is not a facility certification service or a substitute for direct verification.*
