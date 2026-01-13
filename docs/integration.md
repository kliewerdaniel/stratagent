# Frontend-Backend Integration

## Environment Configuration

```env
# .env.local (Next.js)
NEXT_PUBLIC_API_URL=http://localhost:8000

# .env (FastAPI)
DATABASE_URL=sqlite:///./portfolio.db
```

## CORS and Deployment

- **Development**: Backend runs on `http://localhost:8000`, frontend on `http://localhost:3000`
- **Production**:
  - Backend deployed to cloud service (e.g., Railway, Render)
  - Frontend deployed to Vercel with API URL configured
  - CORS configured for production domain

## Data Flow

1. Frontend pages use Server Components to fetch data at build/request time
2. API calls made to backend endpoints
3. Data transformed and displayed using React components
4. SEO metadata generated dynamically from API responses