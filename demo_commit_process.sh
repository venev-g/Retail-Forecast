#!/bin/bash

# Demo Script - Shows automated commit process without pushing
# This script demonstrates the commit automation without actually pushing to GitHub

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}================================${NC}"
echo -e "${PURPLE}AUTOMATED COMMIT DEMO${NC}"
echo -e "${PURPLE}================================${NC}"

echo -e "${BLUE}[INFO]${NC} This is a demonstration of the automated commit process"
echo -e "${BLUE}[INFO]${NC} Repository: Retail-Forecast"
echo -e "${BLUE}[INFO]${NC} Date: $(date)"
echo ""

echo -e "${BLUE}[INFO]${NC} Current git status:"
git status --short

echo ""
echo -e "${BLUE}[INFO]${NC} Files that would be processed:"

# Get list of changed files
CHANGED_FILES=$(git status --porcelain)
COMMIT_COUNT=0

if [[ -z "$CHANGED_FILES" ]]; then
    echo -e "${YELLOW}[WARNING]${NC} No changes detected"
    exit 0
fi

echo "$CHANGED_FILES" | while IFS= read -r line; do
    if [[ -n "$line" ]]; then
        CHANGE_TYPE=$(echo "$line" | cut -c1)
        FILE_PATH=$(echo "$line" | cut -c4-)
        COMMIT_COUNT=$((COMMIT_COUNT + 1))
        
        echo -e "${CYAN}[$COMMIT_COUNT]${NC} $FILE_PATH ($CHANGE_TYPE)"
        
        # Generate commit message based on file
        case "$FILE_PATH" in
            "logging_config.py")
                MSG="feat: add logging configuration module"
                ;;
            "auto_commit_push.sh")
                MSG="feat: add automated git commit and push script"
                ;;
            "auto_commit_push.py")
                MSG="feat: add Python-based automated git commit script"
                ;;
            "CHANGES.md")
                MSG="docs: add comprehensive change documentation"
                ;;
            "GIT_AUTOMATION_README.md")
                MSG="docs: add git automation documentation"
                ;;
            *)
                MSG="chore: update $(basename "$FILE_PATH")"
                ;;
        esac
        
        echo -e "    ${GREEN}Commit message:${NC} $MSG"
        echo ""
    fi
done

TOTAL_FILES=$(echo "$CHANGED_FILES" | wc -l)
echo -e "${GREEN}[SUMMARY]${NC} Would create $TOTAL_FILES individual commits"
echo -e "${BLUE}[INFO]${NC} To actually commit and push, run:"
echo -e "    ${CYAN}./auto_commit_push.sh${NC}"
echo -e "    ${CYAN}./auto_commit_push.py${NC}"

echo ""
echo -e "${PURPLE}================================${NC}"
echo -e "${PURPLE}DEMO COMPLETED${NC}"
echo -e "${PURPLE}================================${NC}"