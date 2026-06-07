#!/usr/bin/env bash
# Run from repo root to start a fresh git history and push to your GitHub.
set -e
git init -b main
git add .
git commit -m "Initial skeleton: monorepo scaffold + architecture README"
echo "Now create an empty repo on GitHub, then:"
echo "  git remote add origin git@github.com:<you>/robo-perception.git"
echo "  git push -u origin main"
