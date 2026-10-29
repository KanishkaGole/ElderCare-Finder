from sqlalchemy import String, Integer, Float, Text, ForeignKey, DateTime, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from .db import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())
    profile: Mapped["Profile"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")

class Profile(Base):
    __tablename__ = "profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    searching_for: Mapped[str] = mapped_column(String(80), default="Parent or family member")
    location: Mapped[str] = mapped_column(String(180), default="")
    care_type: Mapped[str] = mapped_column(String(100), default="Any")
    budget: Mapped[str] = mapped_column(String(80), default="Not sure yet")
    facilities: Mapped[list] = mapped_column(JSON, default=list)
    management: Mapped[str] = mapped_column(String(80), default="Any")
    priority: Mapped[str] = mapped_column(String(100), default="Care suitability")
    gender: Mapped[str] = mapped_column(String(80), default="Any")
    updated_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user: Mapped[User] = relationship(back_populates="profile")

class Home(Base):
    __tablename__ = "homes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    home_name: Mapped[str] = mapped_column(String(300), default="Unnamed listing")
    state: Mapped[str] = mapped_column(String(120), default="")
    district: Mapped[str] = mapped_column(String(120), default="")
    city: Mapped[str] = mapped_column(String(160), default="")
    address: Mapped[str] = mapped_column(Text, default="")
    facility_type: Mapped[str] = mapped_column(String(160), default="")
    care_type: Mapped[str] = mapped_column(String(160), default="")
    gender: Mapped[str] = mapped_column(String(100), default="")
    management: Mapped[str] = mapped_column(String(100), default="")
    capacity: Mapped[float | None] = mapped_column(Float, nullable=True)
    telephone: Mapped[str] = mapped_column(String(300), default="")
    email: Mapped[str] = mapped_column(String(300), default="")
    source: Mapped[str] = mapped_column(String(300), default="")
    search_text: Mapped[str] = mapped_column(Text, default="")
    source_key: Mapped[str] = mapped_column(String(300), unique=True, index=True)

class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    home_id: Mapped[int] = mapped_column(ForeignKey("homes.id", ondelete="CASCADE"), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(30), default="rating")
    context_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reward: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())
