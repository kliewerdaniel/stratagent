# Backend Architecture (Python FastAPI)

## Dependencies

- `fastapi`
- `uvicorn`
- `sqlalchemy`
- `alembic`
- `pydantic`
- `python-multipart` (for file uploads)

## Database Models

```python
# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean
from database import Base

class BlogPost(Base):
    __tablename__ = "blog_posts"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    content = Column(Text, nullable=False)
    excerpt = Column(Text)
    date = Column(DateTime, nullable=False)
    tags = Column(JSON)  # List of strings
    meta_description = Column(String)
    og_image = Column(String)

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=False)
    long_description = Column(Text)
    technologies = Column(JSON)  # List of strings
    image_url = Column(String)
    github_url = Column(String)
    live_url = Column(String)
    featured = Column(Boolean, default=False)

class About(Base):
    __tablename__ = "about"
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    profile_image = Column(String)
```

## API Endpoints

```python
# main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import models, schemas
from database import SessionLocal
from typing import List

app = FastAPI(title="Daniel Kliewer Portfolio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/blog/posts", response_model=List[schemas.BlogPost])
async def get_blog_posts(skip: int = 0, limit: int = 10):
    db = SessionLocal()
    posts = db.query(models.BlogPost).order_by(models.BlogPost.date.desc()).offset(skip).limit(limit).all()
    return posts

@app.get("/api/blog/posts/{slug}", response_model=schemas.BlogPost)
async def get_blog_post(slug: str):
    db = SessionLocal()
    post = db.query(models.BlogPost).filter(models.BlogPost.slug == slug).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

@app.get("/api/projects", response_model=List[schemas.Project])
async def get_projects():
    db = SessionLocal()
    projects = db.query(models.Project).all()
    return projects

@app.get("/api/about", response_model=schemas.About)
async def get_about():
    db = SessionLocal()
    about = db.query(models.About).first()
    return about
```

## Pydantic Schemas

```python
# schemas.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class BlogPostBase(BaseModel):
    title: str
    slug: str
    content: str
    excerpt: Optional[str]
    date: datetime
    tags: List[str]
    meta_description: Optional[str]
    og_image: Optional[str]

class BlogPost(BlogPostBase):
    id: int

    class Config:
        from_attributes = True

class ProjectBase(BaseModel):
    name: str
    slug: str
    description: str
    long_description: Optional[str]
    technologies: List[str]
    image_url: Optional[str]
    github_url: Optional[str]
    live_url: Optional[str]
    featured: bool = False

class Project(ProjectBase):
    id: int

    class Config:
        from_attributes = True

class AboutBase(BaseModel):
    content: str
    profile_image: Optional[str]

class About(AboutBase):
    id: int

    class Config:
        from_attributes = True