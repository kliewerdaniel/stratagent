# Frontend Architecture (Next.js 16+)

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx
│   ├── page.tsx              # Homepage
│   ├── about/
│   │   └── page.tsx
│   ├── blog/
│   │   ├── page.tsx          # Blog listing
│   │   └── [slug]/
│   │       └── page.tsx      # Individual post
│   └── projects/
│       └── page.tsx
├── components/
│   ├── Header.tsx
│   ├── Footer.tsx
│   ├── Hero.tsx
│   ├── BlogCard.tsx
│   ├── ProjectCard.tsx
│   └── SEO.tsx
├── lib/
│   ├── api.ts                # API client functions
│   └── types.ts
└── public/
    └── images/
```

## API Client

```typescript
// lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function getBlogPosts(): Promise<BlogPost[]> {
  const res = await fetch(`${API_BASE}/api/blog/posts`);
  return res.json();
}

export async function getBlogPost(slug: string): Promise<BlogPost> {
  const res = await fetch(`${API_BASE}/api/blog/posts/${slug}`);
  if (!res.ok) throw new Error('Post not found');
  return res.json();
}

export async function getProjects(): Promise<Project[]> {
  const res = await fetch(`${API_BASE}/api/projects`);
  return res.json();
}

export async function getAbout(): Promise<About> {
  const res = await fetch(`${API_BASE}/api/about`);
  return res.json();
}
```

## Types

```typescript
// lib/types.ts
export interface BlogPost {
  id: number;
  title: string;
  slug: string;
  content: string;
  excerpt?: string;
  date: string;
  tags: string[];
  meta_description?: string;
  og_image?: string;
}

export interface Project {
  id: number;
  name: string;
  slug: string;
  description: string;
  long_description?: string;
  technologies: string[];
  image_url?: string;
  github_url?: string;
  live_url?: string;
  featured: boolean;
}

export interface About {
  id: number;
  content: string;
  profile_image?: string;
}
```

## Key Pages

### Homepage (`app/page.tsx`)

```tsx
import { getBlogPosts, getProjects, getAbout } from '@/lib/api';
import Hero from '@/components/Hero';
import FeaturedProjects from '@/components/FeaturedProjects';
import LatestBlogPosts from '@/components/LatestBlogPosts';

export default async function HomePage() {
  const [posts, projects, about] = await Promise.all([
    getBlogPosts(),
    getProjects(),
    getAbout()
  ]);

  return (
    <main>
      <Hero about={about} />
      <FeaturedProjects projects={projects.filter(p => p.featured)} />
      <LatestBlogPosts posts={posts.slice(0, 3)} />
    </main>
  );
}
```

### Blog Listing (`app/blog/page.tsx`)

```tsx
import { getBlogPosts } from '@/lib/api';
import BlogCard from '@/components/BlogCard';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Blog | Daniel Kliewer',
  description: 'Technical blog posts about AI development, autonomous agents, and software engineering',
};

export default async function BlogPage() {
  const posts = await getBlogPosts();

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">Blog</h1>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {posts.map(post => (
          <BlogCard key={post.id} post={post} />
        ))}
      </div>
    </div>
  );
}
```

### Individual Blog Post (`app/blog/[slug]/page.tsx`)

```tsx
import { getBlogPost } from '@/lib/api';
import { notFound } from 'next/navigation';
import { Metadata } from 'next';

interface Props {
  params: { slug: string };
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  try {
    const post = await getBlogPost(params.slug);
    return {
      title: `${post.title} | Daniel Kliewer`,
      description: post.meta_description || post.excerpt,
      openGraph: {
        images: post.og_image ? [{ url: post.og_image }] : [],
      },
    };
  } catch {
    return { title: 'Post Not Found' };
  }
}

export default async function BlogPostPage({ params }: Props) {
  let post;
  try {
    post = await getBlogPost(params.slug);
  } catch {
    notFound();
  }

  return (
    <article className="container mx-auto px-4 py-8 max-w-4xl">
      <header className="mb-8">
        <h1 className="text-4xl font-bold mb-4">{post.title}</h1>
        <div className="text-gray-600 mb-4">
          {new Date(post.date).toLocaleDateString()}
        </div>
        <div className="flex flex-wrap gap-2">
          {post.tags.map(tag => (
            <span key={tag} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
              {tag}
            </span>
          ))}
        </div>
      </header>
      <div
        className="prose max-w-none"
        dangerouslySetInnerHTML={{ __html: post.content }}
      />
    </article>
  );
}
```

## Components

### Header Component

```tsx
// components/Header.tsx
import Link from 'next/link';

export default function Header() {
  return (
    <header className="bg-white shadow-sm">
      <nav className="container mx-auto px-4 py-4">
        <div className="flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold">
            Daniel Kliewer
          </Link>
          <ul className="flex space-x-6">
            <li><Link href="/">Home</Link></li>
            <li><Link href="/about">About</Link></li>
            <li><Link href="/blog">Blog</Link></li>
            <li><Link href="/projects">Projects</Link></li>
          </ul>
        </div>
      </nav>
    </header>
  );
}
```

### BlogCard Component

```tsx
// components/BlogCard.tsx
import Link from 'next/link';
import { BlogPost } from '@/lib/types';

interface Props {
  post: BlogPost;
}

export default function BlogCard({ post }: Props) {
  return (
    <article className="border rounded-lg p-6 hover:shadow-lg transition-shadow">
      <h2 className="text-xl font-semibold mb-2">
        <Link href={`/blog/${post.slug}`} className="hover:text-blue-600">
          {post.title}
        </Link>
      </h2>
      <p className="text-gray-600 mb-4">{post.excerpt}</p>
      <div className="flex justify-between items-center text-sm text-gray-500">
        <span>{new Date(post.date).toLocaleDateString()}</span>
        <div className="flex gap-2">
          {post.tags.slice(0, 3).map(tag => (
            <span key={tag} className="bg-gray-100 px-2 py-1 rounded">
              {tag}
            </span>
          ))}
        </div>
      </div>
    </article>
  );
}