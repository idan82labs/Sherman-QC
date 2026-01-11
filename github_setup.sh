#!/bin/bash
#
# Sherman QC POC - GitHub Setup Script
# Run this locally to create repo and push all code
#
# Usage: ./github_setup.sh
#

set -e

REPO_NAME="sherman-qc-poc"
REPO_DESC="AI-Powered Quality Control System for Sheet Metal Manufacturing - Braude College Final Project 2026"

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║         Sherman QC POC - GitHub Repository Setup                  ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Installing..."
    
    # Detect OS and install
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        sudo apt update && sudo apt install gh
    else
        echo "Please install GitHub CLI manually: https://cli.github.com/"
        exit 1
    fi
fi

echo "✅ GitHub CLI found"
echo ""

# Check authentication
echo "🔐 Checking GitHub authentication..."
if ! gh auth status &> /dev/null; then
    echo "Please authenticate with GitHub:"
    gh auth login
fi
echo "✅ Authenticated with GitHub"
echo ""

# Create repository
echo "📦 Creating repository: $REPO_NAME"
if gh repo view "$REPO_NAME" &> /dev/null 2>&1; then
    echo "⚠️  Repository already exists. Updating..."
else
    gh repo create "$REPO_NAME" \
        --public \
        --description "$REPO_DESC" \
        --source=. \
        --remote=origin \
        --push
    echo "✅ Repository created!"
fi

# Push code
echo ""
echo "🚀 Pushing code to GitHub..."
git push -u origin main --force

# Create release
echo ""
echo "📋 Creating release v2.0.0..."
gh release create v2.0.0 \
    --title "v2.0.0 - AI-Powered Quality Control" \
    --notes-file RELEASE_NOTES.md \
    --latest

# Get repo URL
REPO_URL=$(gh repo view --json url -q .url)

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                        ✅ SUCCESS!                                ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║                                                                   ║"
echo "║  Repository: $REPO_URL"
echo "║                                                                   ║"
echo "║  Release: $REPO_URL/releases/tag/v2.0.0"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 Sherman QC POC is now live on GitHub!"
