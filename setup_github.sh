#!/bin/bash
# GitHub Repository Setup Script
# Run this after creating a repo on GitHub

# Configuration - UPDATE THESE
GITHUB_USER="your-username"
REPO_NAME="scan-qc-system"

echo "================================================"
echo "  Scan QC System - GitHub Setup"
echo "================================================"
echo ""
echo "Step 1: Create a new repository on GitHub"
echo "  - Go to: https://github.com/new"
echo "  - Repository name: $REPO_NAME"
echo "  - Description: AI-Powered Quality Control for Sheet Metal Parts"
echo "  - Set to Private or Public as needed"
echo "  - DO NOT initialize with README (we have one)"
echo ""
echo "Step 2: After creating the repo, run these commands:"
echo ""
echo "  cd scan_qc_app"
echo "  git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
echo "  git push -u origin main"
echo ""
echo "Or with SSH:"
echo "  git remote add origin git@github.com:$GITHUB_USER/$REPO_NAME.git"
echo "  git push -u origin main"
echo ""
echo "================================================"

# Uncomment below after updating GITHUB_USER
# git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git
# git push -u origin main
