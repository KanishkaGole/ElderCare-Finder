from pathlib import Path
import os, re, json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .db import Base, engine, get_db, SessionLocal
from .models import User, Profile, Home, Feedback
from .security import hash_password, verify_password, create_token, current_user

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("HOMES_XLSX", str(ROOT / "data" / "All_India_Old_Age_Homes.xlsx")))
app = FastAPI(title="ElderCare Finder API", version="1.0.0", description="Personalized old-age-home discovery with content-based RS and a contextual-bandit prototype")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
class LoginIn(BaseModel):
    email: EmailStr
    password: str
class ProfileIn(BaseModel):
    searching_for: str = "Parent or family member"
    location: str = ""
    care_type: str = "Any"
    budget: str = "Not sure yet"
    facilities: list[str] = Field(default_factory=list)
    management: str = "Any"
    priority: str = "Care suitability"
    gender: str = "Any"
class RecommendIn(BaseModel):
    query: str = ""
    state: str = "Any"
    district: str = "Any"
    facility_type: str = "Any"
    care_type: str = "Any"
    gender: str = "Any"
    management: str = "Any"
    top_n: int = Field(default=10, ge=1, le=30)
    adaptive: bool = True
class FeedbackIn(BaseModel):
    home_id: int
    rating: int = Field(ge=1, le=5)
    action: str = "rating"
    context: dict = Field(default_factory=dict)

Base.metadata.create_all(bind=engine)

def clean(v):
    if v is None or pd.isna(v): return ""
    return re.sub(r"\s+", " ", str(v)).strip()

def seed_homes(db: Session):
    if db.scalar(select(func.count(Home.id))) > 0: return
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}; place All_India_Old_Age_Homes.xlsx in data/")
        return
    frame = pd.read_excel(DATA_PATH, sheet_name="Master_Data")
    for i, r in frame.iterrows():
        name = clean(r.get("Home Name")) or "Unnamed listing"
        state, district, city = clean(r.get("State")), clean(r.get("District")), clean(r.get("City / Taluka"))
        address, facility, care = clean(r.get("Address")), clean(r.get("Facility Type")), clean(r.get("Care Type"))
        gender, mgmt = clean(r.get("Gender")), clean(r.get("Mgmt"))
        phone, email, source = clean(r.get("Telephone")), clean(r.get("Email")), clean(r.get("Source"))
        cap = pd.to_numeric(r.get("Bed Capacity"), errors="coerce")
        text = " ".join([name, state, district, city, address, facility, care, gender, mgmt, clean(r.get("Remarks"))])
        key = f"{i}:{name}:{state}:{district}"
        db.add(Home(home_name=name, state=state, district=district, city=city, address=address, facility_type=facility, care_type=care, gender=gender, management=mgmt, capacity=float(cap) if pd.notna(cap) else None, telephone=phone, email=email, source=source, search_text=text, source_key=key))
    db.commit()

@app.on_event("startup")
def startup():
    with SessionLocal() as db: seed_homes(db)

def profile_dict(p):
    if not p: return None
    return {"searching_for":p.searching_for,"location":p.location,"care_type":p.care_type,"budget":p.budget,"facilities":p.facilities or [],"management":p.management,"priority":p.priority,"gender":p.gender}
def user_dict(u): return {"id":u.id,"name":u.name,"email":u.email,"profile":profile_dict(u.profile)}

@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    return {"status":"ok", "homes":db.scalar(select(func.count(Home.id))) or 0, "service":"ElderCare Finder"}

@app.post("/api/auth/register")
def register(body: RegisterIn, db: Session = Depends(get_db)):
    email = body.email.lower().strip()
    if db.scalar(select(User).where(User.email == email)): raise HTTPException(409, "An account with this email already exists")
    u = User(name=body.name.strip(), email=email, password_hash=hash_password(body.password))
    db.add(u); db.commit(); db.refresh(u)
    return {"access_token":create_token(u.id),"token_type":"bearer","user":user_dict(u)}

@app.post("/api/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.email == body.email.lower().strip()))
    if not u or not verify_password(body.password, u.password_hash): raise HTTPException(401, "Email or password is incorrect")
    return {"access_token":create_token(u.id),"token_type":"bearer","user":user_dict(u)}

@app.get("/api/me")
def me(user: User = Depends(current_user)): return user_dict(user)

@app.put("/api/profile")
def save_profile(body: ProfileIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = db.scalar(select(Profile).where(Profile.user_id == user.id))
    if not p: p = Profile(user_id=user.id); db.add(p)
    for field, value in body.model_dump().items(): setattr(p, field, value)
    db.commit(); db.refresh(p); db.refresh(user)
    return {"profile":profile_dict(p),"message":"Preferences saved"}

@app.get("/api/options")
def options(db: Session = Depends(get_db)):
    homes = db.scalars(select(Home)).all()
    def opts(field): return sorted({getattr(h,field) for h in homes if getattr(h,field) and getattr(h,field).strip()})
    return {"states":opts("state"),"districts":opts("district"),"facility_types":opts("facility_type"),"care_types":opts("care_type"),"management_types":opts("management"),"genders":opts("gender")}

def bandit_state(user_id):
    # Per-user linear contextual bandit, persisted to a small JSON file.
    path = Path(__file__).resolve().parents[1] / "artifacts" / f"bandit_user_{user_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    d=7
    if path.exists():
        try:
            data=json.loads(path.read_text()); return np.array(data["A"],float),np.array(data["b"],float),int(data.get("updates",0)),path
        except Exception: pass
    return np.eye(d), np.zeros(d), 0, path

def home_features(h, sim, query):
    return np.array([1.0, sim, float(bool(h.telephone)), float(bool(h.email)), float(bool(h.address)), float(h.capacity is not None and h.capacity > 0), float(bool(query.strip()))])

def make_candidates(db, body, user, profile):
    homes = db.scalars(select(Home)).all()
    f = profile_dict(profile) or {}
    # Explicit request filters override profile preferences; questionnaire location acts as a soft query unless a state is selected.
    state = body.state if body.state != "Any" else "Any"
    district = body.district if body.district != "Any" else "Any"
    facility = body.facility_type if body.facility_type != "Any" else "Any"
    care = body.care_type if body.care_type != "Any" else (f.get("care_type") or "Any")
    gender = body.gender if body.gender != "Any" else (f.get("gender") or "Any")
    mgmt = body.management if body.management != "Any" else (f.get("management") or "Any")
    query_parts = [body.query, f.get("location", ""), " ".join(f.get("facilities", []))]
    if f.get("priority"): query_parts.append(f["priority"])
    query = " ".join(x for x in query_parts if x and x not in {"Any", "Not sure yet"})
    eligible=[]
    for h in homes:
        if state != "Any" and h.state.casefold()!=state.casefold(): continue
        if district != "Any" and h.district.casefold()!=district.casefold(): continue
        if facility != "Any" and h.facility_type.casefold()!=facility.casefold(): continue
        if care not in {"Any", "Not sure yet"} and care.casefold() not in h.care_type.casefold(): continue
        if gender in {"Men","Women"}:
            g=h.gender.casefold().replace("woman", "women").casefold()
            if g not in {gender.casefold(), "men & women", "men & women"}: continue
        if mgmt != "Any":
            aliases = {"government-run": ["govt", "government", "swd"], "ngo or charitable": ["ngo"], "private": ["pvt", "private"], "public-private partnership": ["ppp"]}
            accepted = aliases.get(mgmt.casefold(), [mgmt.casefold()])
            if not any(a in h.management.casefold() for a in accepted): continue
        eligible.append(h)
    if not eligible: return [], query
    texts=[h.search_text for h in eligible]
    vec=TfidfVectorizer(ngram_range=(1,2), max_features=8000, stop_words="english")
    matrix=vec.fit_transform(texts)
    if query.strip():
        qv=vec.transform([query]); sims=cosine_similarity(qv,matrix).ravel()
    else: sims=np.zeros(len(eligible))
    scored=[]
    for h, sim in zip(eligible,sims):
        contact=(0.5*bool(h.telephone)+0.2*bool(h.email)+0.3*bool(h.address))
        base=float(sim)*0.85+contact*0.15
        x=home_features(h,float(sim),query)
        scored.append({"home":h,"similarity":float(sim),"contact":float(contact),"base_score":base,"x":x})
    if body.adaptive:
        A,b,updates,path=bandit_state(user.id); inv=np.linalg.pinv(A); theta=inv@b
        for item in scored:
            x=item["x"]; pred=float(x@theta); bonus=0.08*float(np.sqrt(max(0,x@inv@x)))
            item["rl_score"]=pred+bonus
            item["final_score"]=0.65*item["base_score"]+0.35*(1/(1+np.exp(-np.clip(item["rl_score"],-10,10))))
        scored.sort(key=lambda x:(x["final_score"],x["base_score"]),reverse=True)
    else:
        for item in scored: item["rl_score"]=None; item["final_score"]=item["base_score"]
        scored.sort(key=lambda x:x["final_score"],reverse=True)
    return scored[:body.top_n], query

@app.post("/api/recommendations")
def recommendations(body: RecommendIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    profile=db.scalar(select(Profile).where(Profile.user_id==user.id))
    scored, query=make_candidates(db,body,user,profile)
    results=[]
    for item in scored:
        h=item["home"]
        results.append({"id":h.id,"home_name":h.home_name,"state":h.state,"district":h.district,"city":h.city,"address":h.address,"facility_type":h.facility_type or "Not listed","care_type":h.care_type or "Not listed","gender":h.gender or "Not listed","management":h.management or "Not listed","capacity":h.capacity,"telephone":h.telephone or "","email":h.email or "","source":h.source or "","similarity_score":round(item["similarity"],4),"match_score":round(item["final_score"],4),"rl_score":None if item["rl_score"] is None else round(item["rl_score"],4),"reason":reason_for(h,profile,item),"warnings":warnings_for(h)})
    return {"results":results,"query":query,"count":len(results),"adaptive":body.adaptive,"message":"No homes match all selected filters. Try widening location or optional preferences." if not results else None}

def reason_for(h, profile, item):
    reasons=[]
    if item["similarity"]>0.05: reasons.append("matches your search terms")
    if profile:
        if profile.location and any(x.casefold() in (h.state+" "+h.district+" "+h.city).casefold() for x in [profile.location]): reasons.append("matches your preferred area")
        if profile.care_type not in ("Any","Not sure yet") and profile.care_type.casefold() in h.care_type.casefold(): reasons.append("matches your care preference")
        if profile.facilities: reasons.append("ranked using your selected facility interests; verify services directly")
    if h.telephone or h.email: reasons.append("has contact information listed")
    return reasons[:3] or ["matches your mandatory filters; details require confirmation"]

def warnings_for(h):
    w=[]
    if not h.telephone and not h.email: w.append("No phone or email listed in this record")
    if h.capacity is None or h.capacity<=0: w.append("Bed capacity missing or zero; confirm directly")
    w.append("Directory information may be outdated; confirm fees, availability, services and eligibility directly")
    return w

@app.post("/api/feedback")
def feedback(body: FeedbackIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    h=db.get(Home,body.home_id)
    if not h: raise HTTPException(404,"Home not found")
    reward=(body.rating-3)/2
    event=Feedback(user_id=user.id,home_id=h.id,rating=body.rating,action=body.action[:30],context_json=body.context,reward=reward)
    db.add(event); db.commit()
    # Update a per-user linear bandit using listing features. This is a contextual-bandit prototype, not long-horizon MDP RL.
    x=home_features(h, float(body.context.get("similarity_score",0)), str(body.context.get("query","")))
    A,b,updates,path=bandit_state(user.id); A+=np.outer(x,x); b+=reward*x
    path.write_text(json.dumps({"A":A.tolist(),"b":b.tolist(),"updates":updates+1}))
    return {"message":"Feedback saved. Your adaptive ranking has been updated.","updates":updates+1,"reward":reward}

@app.get("/api/metrics")
def metrics(db: Session = Depends(get_db), user: User = Depends(current_user)):
    total=db.scalar(select(func.count(Feedback.id)).where(Feedback.user_id==user.id)) or 0
    avg=db.scalar(select(func.avg(Feedback.rating)).where(Feedback.user_id==user.id))
    _,_,updates,_=bandit_state(user.id)
    return {"feedback_count":total,"average_rating":round(float(avg),2) if avg is not None else None,"policy_updates":updates,"homes_in_directory":db.scalar(select(func.count(Home.id))) or 0}
