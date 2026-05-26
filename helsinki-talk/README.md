# Imprecision in control problems — Beyond Basic Bayesianism, Helsinki, 26 May 2026

Self-contained slide deck (HTML + assets) for hosting on GitHub Pages.

## Deploy

1. Create a new public GitHub repository.
2. Upload the contents of this folder to the repository root.
3. In the repository Settings → Pages, set Source to "Deploy from a branch"
   and select the `main` branch with folder `/ (root)`.
4. Wait for the deployment, then open the URL GitHub Pages provides.

The deck loads MathJax from a CDN. No build step is required.

## Local preview

Run a static HTTP server from this folder, e.g.:

    python3 -m http.server 8000

Then open `http://localhost:8000`.
