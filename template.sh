#!/bin/bash
# ─────────────────────────────────────────────
# MediAssist — Project Scaffolding Script
# Creates the full directory structure and files
# ─────────────────────────────────────────────

# Creating directories
mkdir -p src
mkdir -p data
mkdir -p static
mkdir -p templates
mkdir -p research

# Creating source files
touch src/__init__.py
touch src/helper.py
touch src/prompt.py
touch src/store_csv_index.py

# Creating app files
touch app.py
touch user_db.py
touch store_index.py
touch setup.py

# Creating template files
touch templates/chat.html
touch templates/login.html
touch templates/register.html

# Creating static files
touch static/style.css

# Creating config files
touch .env
touch .gitignore
touch .dockerignore
touch Dockerfile
touch requirements.txt
touch README.md
touch LICENSE

# Creating research files
touch research/trials.ipynb

echo "✅ MediAssist project structure created successfully!"
echo ""
echo "📁 Next steps:"
echo "  1. Add your API keys to .env"
echo "  2. Add medical PDFs to data/"
echo "  3. Run: pip install -r requirements.txt"
echo "  4. Run: python store_index.py"
echo "  5. Run: python app.py"
