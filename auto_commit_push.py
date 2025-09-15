#!/usr/bin/env python3
"""
Automated Git Commit and Push Script for Retail-Forecast Pipeline
This script analyzes changes, generates commit messages, and pushes to GitHub
"""

import subprocess
import sys
import os
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class GitAutomator:
    def __init__(self, repo_name="Retail-Forecast", branch="main"):
        self.repo_name = repo_name
        self.branch = branch
        self.commit_count = 0
        
    def print_status(self, message):
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
        
    def print_success(self, message):
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
        
    def print_warning(self, message):
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
        
    def print_error(self, message):
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
        
    def print_header(self, message):
        print(f"{Colors.PURPLE}{'=' * 50}{Colors.NC}")
        print(f"{Colors.PURPLE}{message}{Colors.NC}")
        print(f"{Colors.PURPLE}{'=' * 50}{Colors.NC}")

    def run_git_command(self, cmd, capture_output=True, check=True):
        """Run a git command and return the result"""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=capture_output, 
                text=True, 
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            if not check:
                return e
            self.print_error(f"Git command failed: {cmd}")
            self.print_error(f"Error: {e.stderr if e.stderr else str(e)}")
            return None

    def generate_commit_message(self, file_path, change_type):
        """Generate appropriate commit message based on file and change type"""
        
        commit_messages = {
            "logging_config.py": """feat: add logging configuration module

- Create centralized logging configuration
- Support console and file output
- Configure log levels for different modules
- Fix import error for configure_logging function""",
            
            "configs/training.yaml": """fix: update ZenML training pipeline configuration

- Fix step configuration validation errors
- Update YAML structure for ZenML compatibility
- Add proper parameters nesting
- Replace comments with empty dictionaries""",
            
            "configs/inference.yaml": """fix: update ZenML inference pipeline configuration

- Fix step configuration validation errors
- Update YAML structure for ZenML compatibility
- Add proper parameters nesting
- Replace comments with empty dictionaries""",
            
            "materializers/prophet_materializer.py": """fix: resolve Prophet model materializer issues

- Fix type comparison using 'is' instead of '=='
- Implement safe filename generation for model storage
- Use proper artifact store methods for file operations
- Add error handling for model loading/saving""",
            
            "steps/data_visualizer.py": """fix: resolve Content Security Policy violations in visualizations

- Remove external CDN script loading
- Use include_plotlyjs=True for self-contained HTML
- Make visualizations CSP-compliant
- Ensure offline functionality for charts""",
            
            "CHANGES.md": """docs: add comprehensive change documentation

- Document all pipeline fixes and improvements
- Include code changes and configuration updates
- Add troubleshooting guide and commands
- Provide complete technical reference""",
            
            "requirements.txt": """deps: update project dependencies

- Add missing package requirements
- Update version constraints
- Ensure compatibility with ZenML and Prophet""",
            
            "auto_commit_push.sh": """feat: add automated git commit and push script

- Analyze repository changes automatically
- Generate contextual commit messages
- Create individual commits per logical change
- Support dry-run and verbose modes""",
            
            "auto_commit_push.py": """feat: add Python-based automated git commit script

- Alternative Python implementation for git automation
- Object-oriented design with error handling
- Cross-platform compatibility
- Comprehensive logging and status reporting"""
        }
        
        # Check for exact file match
        if file_path in commit_messages:
            return commit_messages[file_path]
        
        # Generate generic message based on change type and file
        filename = os.path.basename(file_path)
        
        if change_type == 'A':  # Added
            return f"""feat: add {filename}

- Implement new functionality
- Add required module/configuration"""
        elif change_type == 'M':  # Modified
            return f"""fix: update {filename}

- Apply fixes and improvements
- Resolve compatibility issues
- Enhance functionality"""
        elif change_type == 'D':  # Deleted
            return f"""remove: delete {filename}

- Clean up obsolete files
- Remove unused components"""
        else:
            return f"""chore: update {filename}

- Apply general updates and improvements"""

    def check_git_setup(self):
        """Check if git is properly configured"""
        self.print_status("Checking git configuration...")
        
        # Check if we're in a git repository
        result = self.run_git_command("git status", check=False)
        if result.returncode != 0:
            self.print_error("Not a git repository. Please initialize git first.")
            return False
        
        # Check if user is configured
        result = self.run_git_command("git config user.name", check=False)
        if result.returncode != 0 or not result.stdout.strip():
            self.print_error("Git user name not configured.")
            self.print_status("Please configure git user:")
            print("git config --global user.name 'Your Name'")
            print("git config --global user.email 'your.email@example.com'")
            return False
        
        self.print_success("Git configuration OK")
        return True

    def get_changed_files(self):
        """Get list of changed files with their status"""
        result = self.run_git_command("git status --porcelain")
        if not result or not result.stdout.strip():
            return []
        
        changed_files = []
        for line in result.stdout.strip().split('\n'):
            if line:
                change_type = line[0]
                file_path = line[3:]
                changed_files.append((change_type, file_path))
        
        return changed_files

    def commit_single_file(self, file_path, change_type):
        """Stage and commit a single file"""
        self.print_status(f"Processing: {file_path} ({change_type})")
        
        # Stage the file
        stage_result = self.run_git_command(f"git add '{file_path}'")
        if not stage_result:
            return False
        
        # Generate commit message
        commit_msg = self.generate_commit_message(file_path, change_type)
        
        # Create commit
        commit_result = self.run_git_command(f"git commit -m '{commit_msg}'", check=False)
        if commit_result.returncode == 0:
            self.commit_count += 1
            self.print_success(f"Committed: {file_path}")
            # Show first line of commit message
            first_line = commit_msg.split('\n')[0]
            print(f"{Colors.CYAN}Commit #{self.commit_count}:{Colors.NC} {first_line}")
            return True
        else:
            self.print_warning(f"No changes to commit for: {file_path}")
            return False

    def process_changes(self):
        """Process all changes in the repository"""
        self.print_header("ANALYZING REPOSITORY CHANGES")
        
        changed_files = self.get_changed_files()
        if not changed_files:
            self.print_warning("No changes detected in the repository")
            return 0
        
        self.print_status(f"Found {len(changed_files)} changed files. Processing...")
        
        successful_commits = 0
        for change_type, file_path in changed_files:
            # Skip if file doesn't exist (in case of deletions)
            if change_type != 'D' and not os.path.exists(file_path):
                continue
            
            if self.commit_single_file(file_path, change_type):
                successful_commits += 1
        
        return successful_commits

    def push_to_github(self):
        """Push changes to GitHub"""
        self.print_header("PUSHING CHANGES TO GITHUB")
        
        if self.commit_count == 0:
            self.print_warning("No commits to push")
            return True
        
        self.print_status(f"Pushing {self.commit_count} commits to GitHub...")
        
        # Try regular push first
        push_result = self.run_git_command(f"git push origin {self.branch}", check=False)
        if push_result.returncode == 0:
            self.print_success(f"Successfully pushed {self.commit_count} commits to GitHub!")
            return True
        
        # Try with upstream setting
        self.print_warning("Push failed. Trying to set upstream...")
        push_upstream_result = self.run_git_command(f"git push -u origin {self.branch}", check=False)
        if push_upstream_result.returncode == 0:
            self.print_success(f"Successfully pushed {self.commit_count} commits to GitHub!")
            return True
        
        self.print_error("Failed to push to GitHub.")
        self.print_status("You may need to:")
        print("1. Check your GitHub authentication (token/SSH key)")
        print("2. Verify repository permissions")
        print("3. Ensure the remote repository exists")
        return False

    def show_summary(self):
        """Show summary of operations"""
        self.print_header("COMMIT SUMMARY")
        
        if self.commit_count > 0:
            self.print_success(f"Total commits created: {self.commit_count}")
            print()
            self.print_status("Recent commits:")
            result = self.run_git_command(f"git log --oneline -n {self.commit_count}")
            if result:
                print(result.stdout)
        else:
            self.print_warning("No commits were created")
        
        print()
        self.print_status("Repository status:")
        result = self.run_git_command("git status --short")
        if result:
            print(result.stdout)

    def dry_run(self):
        """Show what would be committed without making changes"""
        self.print_header("DRY RUN - PREVIEW OF CHANGES")
        
        changed_files = self.get_changed_files()
        if not changed_files:
            self.print_warning("No changes detected in the repository")
            return
        
        for i, (change_type, file_path) in enumerate(changed_files, 1):
            print(f"{Colors.YELLOW}[{i}]{Colors.NC} {file_path} ({change_type})")
            
            commit_msg = self.generate_commit_message(file_path, change_type)
            first_line = commit_msg.split('\n')[0]
            print(f"{Colors.CYAN}Commit message:{Colors.NC} {first_line}")
            print()
        
        self.print_status(f"Would create {len(changed_files)} commits for the above files")
        self.print_warning("No actual commits made (dry run mode)")

    def run(self):
        """Main execution function"""
        self.print_header("RETAIL FORECAST PIPELINE - AUTOMATED GIT COMMIT & PUSH")
        print(f"{Colors.CYAN}Repository:{Colors.NC} {self.repo_name}")
        print(f"{Colors.CYAN}Branch:{Colors.NC} {self.branch}")
        print(f"{Colors.CYAN}Date:{Colors.NC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check prerequisites
        if not self.check_git_setup():
            return False
        
        # Process all changes
        successful_commits = self.process_changes()
        
        # Push to GitHub if we have commits
        if successful_commits > 0:
            if not self.push_to_github():
                return False
        
        # Show summary
        self.show_summary()
        
        self.print_success("Automated commit and push completed!")
        return True


def show_help():
    """Show help message"""
    print("Automated Git Commit and Push Script (Python)")
    print()
    print(f"Usage: {sys.argv[0]} [OPTIONS]")
    print()
    print("Options:")
    print("  -h, --help     Show this help message")
    print("  -n, --dry-run  Show what would be committed without making changes")
    print("  -v, --verbose  Enable verbose output")
    print()
    print("This script will:")
    print("1. Analyze all changed files in the repository")
    print("2. Generate appropriate commit messages for each file")
    print("3. Create individual commits for each logical change")
    print("4. Push all commits to GitHub")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ['-h', '--help']:
            show_help()
            return
        elif arg in ['-n', '--dry-run']:
            automator = GitAutomator()
            automator.dry_run()
            return
        elif arg in ['-v', '--verbose']:
            # Enable verbose mode (could add more detailed logging)
            pass
        else:
            print(f"Unknown option: {arg}")
            show_help()
            return
    
    # Run the main automation
    automator = GitAutomator()
    success = automator.run()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()