#!/bin/bash

# Automated Git Commit and Push Script for Retail-Forecast Pipeline
# This script analyzes changes, generates commit messages, and pushes to GitHub

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="Retail-Forecast"
BRANCH_NAME="main"
COMMIT_COUNT=0

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# Function to generate commit message based on file changes
generate_commit_message() {
    local file_path="$1"
    local change_type="$2"
    
    case "$file_path" in
        "logging_config.py")
            echo "feat: add logging configuration module

- Create centralized logging configuration
- Support console and file output
- Configure log levels for different modules
- Fix import error for configure_logging function"
            ;;
        "configs/training.yaml"|"configs/inference.yaml")
            echo "fix: update ZenML pipeline configuration format

- Fix step configuration validation errors
- Update YAML structure for ZenML compatibility
- Add proper parameters nesting
- Replace comments with empty dictionaries"
            ;;
        "materializers/prophet_materializer.py")
            echo "fix: resolve Prophet model materializer issues

- Fix type comparison using 'is' instead of '=='
- Implement safe filename generation for model storage
- Use proper artifact store methods for file operations
- Add error handling for model loading/saving"
            ;;
        "steps/data_visualizer.py")
            echo "fix: resolve Content Security Policy violations in visualizations

- Remove external CDN script loading
- Use include_plotlyjs=True for self-contained HTML
- Make visualizations CSP-compliant
- Ensure offline functionality for charts"
            ;;
        "CHANGES.md")
            echo "docs: add comprehensive change documentation

- Document all pipeline fixes and improvements
- Include code changes and configuration updates
- Add troubleshooting guide and commands
- Provide complete technical reference"
            ;;
        "requirements.txt")
            echo "deps: update project dependencies

- Add missing package requirements
- Update version constraints
- Ensure compatibility with ZenML and Prophet"
            ;;
        *)
            # Generic commit message based on change type
            local filename=$(basename "$file_path")
            case "$change_type" in
                "M")
                    echo "fix: update $filename

- Apply fixes and improvements
- Resolve compatibility issues
- Enhance functionality"
                    ;;
                "A")
                    echo "feat: add $filename

- Implement new functionality
- Add required module/configuration"
                    ;;
                "D")
                    echo "remove: delete $filename

- Clean up obsolete files
- Remove unused components"
                    ;;
                *)
                    echo "chore: update $filename

- Apply general updates and improvements"
                    ;;
            esac
            ;;
    esac
}

# Function to check if git is initialized and configured
check_git_setup() {
    print_status "Checking git configuration..."
    
    if ! git status >/dev/null 2>&1; then
        print_error "Not a git repository. Initializing..."
        git init
        git remote add origin "https://github.com/venev-g/${REPO_NAME}.git" 2>/dev/null || true
    fi
    
    # Check if user is configured
    if ! git config user.name >/dev/null 2>&1; then
        print_warning "Git user name not configured. Please set it:"
        echo "git config --global user.name 'Your Name'"
        echo "git config --global user.email 'your.email@example.com'"
        exit 1
    fi
    
    print_success "Git configuration OK"
}

# Function to stage and commit a single file
commit_single_file() {
    local file_path="$1"
    local change_type="$2"
    
    print_status "Processing: $file_path ($change_type)"
    
    # Stage the file
    git add "$file_path"
    
    # Generate commit message
    local commit_msg=$(generate_commit_message "$file_path" "$change_type")
    
    # Create commit
    if git commit -m "$commit_msg" >/dev/null; then
        COMMIT_COUNT=$((COMMIT_COUNT + 1))
        print_success "Committed: $file_path"
        echo -e "${CYAN}Commit #${COMMIT_COUNT}:${NC} $(echo "$commit_msg" | head -n1)"
    else
        print_warning "No changes to commit for: $file_path"
    fi
}

# Function to process all changes
process_changes() {
    print_header "ANALYZING REPOSITORY CHANGES"
    
    # Check if there are any changes
    if ! git status --porcelain | grep -q .; then
        print_warning "No changes detected in the repository"
        return 0
    fi
    
    print_status "Found changes. Processing individual files..."
    
    # Get list of changed files with their status
    local changed_files=$(git status --porcelain)
    
    echo "$changed_files" | while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            local change_type=$(echo "$line" | cut -c1)
            local file_path=$(echo "$line" | cut -c4-)
            
            # Skip if file doesn't exist (in case of deletions)
            if [[ "$change_type" != "D" ]] && [[ ! -f "$file_path" ]]; then
                continue
            fi
            
            commit_single_file "$file_path" "$change_type"
        fi
    done
    
    # Update commit count from subshell
    COMMIT_COUNT=$(git rev-list --count HEAD 2>/dev/null || echo "0")
}

# Function to push changes to GitHub
push_to_github() {
    print_header "PUSHING CHANGES TO GITHUB"
    
    if [[ $COMMIT_COUNT -eq 0 ]]; then
        print_warning "No commits to push"
        return 0
    fi
    
    print_status "Pushing $COMMIT_COUNT commits to GitHub..."
    
    # Try to push
    if git push origin "$BRANCH_NAME" 2>/dev/null; then
        print_success "Successfully pushed $COMMIT_COUNT commits to GitHub!"
    else
        print_warning "Push failed. Trying to set upstream..."
        if git push -u origin "$BRANCH_NAME"; then
            print_success "Successfully pushed $COMMIT_COUNT commits to GitHub!"
        else
            print_error "Failed to push to GitHub. Please check your credentials and repository access."
            print_status "You may need to:"
            echo "1. Check your GitHub authentication (token/SSH key)"
            echo "2. Verify repository permissions"
            echo "3. Ensure the remote repository exists"
            return 1
        fi
    fi
}

# Function to show summary
show_summary() {
    print_header "COMMIT SUMMARY"
    
    if [[ $COMMIT_COUNT -gt 0 ]]; then
        print_success "Total commits created: $COMMIT_COUNT"
        echo ""
        print_status "Recent commits:"
        git log --oneline -n $COMMIT_COUNT
    else
        print_warning "No commits were created"
    fi
    
    echo ""
    print_status "Repository status:"
    git status --short
}

# Main execution function
main() {
    print_header "RETAIL FORECAST PIPELINE - AUTOMATED GIT COMMIT & PUSH"
    echo -e "${CYAN}Repository:${NC} $REPO_NAME"
    echo -e "${CYAN}Branch:${NC} $BRANCH_NAME"
    echo -e "${CYAN}Date:${NC} $(date)"
    echo ""
    
    # Check prerequisites
    check_git_setup
    
    # Process all changes
    process_changes
    
    # Update commit count after processing
    local total_commits=$(git log --oneline 2>/dev/null | wc -l)
    COMMIT_COUNT=$total_commits
    
    # Push to GitHub
    push_to_github
    
    # Show summary
    show_summary
    
    print_success "Automated commit and push completed!"
}

# Help function
show_help() {
    echo "Automated Git Commit and Push Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -n, --dry-run  Show what would be committed without making changes"
    echo "  -v, --verbose  Enable verbose output"
    echo ""
    echo "This script will:"
    echo "1. Analyze all changed files in the repository"
    echo "2. Generate appropriate commit messages for each file"
    echo "3. Create individual commits for each logical change"
    echo "4. Push all commits to GitHub"
    echo ""
    echo "Example:"
    echo "  $0                 # Commit and push all changes"
    echo "  $0 --dry-run       # Preview what would be committed"
}

# Dry run function
dry_run() {
    print_header "DRY RUN - PREVIEW OF CHANGES"
    
    if ! git status --porcelain | grep -q .; then
        print_warning "No changes detected in the repository"
        return 0
    fi
    
    local changed_files=$(git status --porcelain)
    local preview_count=0
    
    echo "$changed_files" | while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            local change_type=$(echo "$line" | cut -c1)
            local file_path=$(echo "$line" | cut -c4-)
            
            preview_count=$((preview_count + 1))
            echo -e "${YELLOW}[$preview_count]${NC} $file_path ($change_type)"
            
            local commit_msg=$(generate_commit_message "$file_path" "$change_type")
            echo -e "${CYAN}Commit message:${NC} $(echo "$commit_msg" | head -n1)"
            echo ""
        fi
    done
    
    print_status "Would create commits for the above files"
    print_warning "No actual commits made (dry run mode)"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -n|--dry-run)
        dry_run
        exit 0
        ;;
    -v|--verbose)
        set -x
        main
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac