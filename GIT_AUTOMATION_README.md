# Automated Git Commit and Push Scripts

This repository includes two automated scripts to commit and push changes to GitHub with intelligent commit message generation.

## Scripts Available

### 1. `auto_commit_push.sh` (Bash Script)
A comprehensive bash script with full featured automation.

### 2. `auto_commit_push.py` (Python Script)
A Python implementation with object-oriented design and cross-platform compatibility.

## Features

- **Automatic Change Detection**: Analyzes all modified, added, and deleted files
- **Intelligent Commit Messages**: Generates contextual commit messages based on file types and changes
- **Individual Commits**: Creates separate commits for each logical change
- **Commit Counting**: Tracks and reports total number of commits created
- **GitHub Push**: Automatically pushes all commits to the remote repository
- **Dry Run Mode**: Preview changes without committing
- **Verbose Output**: Detailed logging and status reporting
- **Error Handling**: Comprehensive error checking and recovery

## Usage

### Basic Usage

```bash
# Using bash script
./auto_commit_push.sh

# Using Python script
./auto_commit_push.py
# or
python3 auto_commit_push.py
```

### Options

```bash
# Show help
./auto_commit_push.sh --help
./auto_commit_push.py --help

# Dry run (preview changes without committing)
./auto_commit_push.sh --dry-run
./auto_commit_push.py --dry-run

# Verbose mode
./auto_commit_push.sh --verbose
./auto_commit_push.py --verbose
```

## Commit Message Generation

The scripts automatically generate appropriate commit messages based on the files being changed:

| File/Pattern | Commit Type | Example Message |
|-------------|-------------|----------------|
| `logging_config.py` | `feat:` | "feat: add logging configuration module" |
| `configs/*.yaml` | `fix:` | "fix: update ZenML pipeline configuration format" |
| `materializers/*.py` | `fix:` | "fix: resolve Prophet model materializer issues" |
| `steps/data_visualizer.py` | `fix:` | "fix: resolve Content Security Policy violations" |
| `CHANGES.md` | `docs:` | "docs: add comprehensive change documentation" |
| `requirements.txt` | `deps:` | "deps: update project dependencies" |
| New files | `feat:` | "feat: add [filename]" |
| Modified files | `fix:` | "fix: update [filename]" |
| Deleted files | `remove:` | "remove: delete [filename]" |

## Prerequisites

### For Both Scripts
- Git repository initialized
- Git user configuration set:
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your.email@example.com"
  ```
- GitHub remote repository configured
- Authentication set up (SSH key or personal access token)

### For Python Script Only
- Python 3.6 or higher

## Script Workflow

1. **Setup Check**: Verifies git configuration and repository status
2. **Change Detection**: Identifies all modified, added, and deleted files
3. **File Processing**: For each changed file:
   - Stages the file
   - Generates appropriate commit message
   - Creates individual commit
4. **Push to GitHub**: Pushes all commits to the remote repository
5. **Summary Report**: Shows total commits created and repository status

## Examples

### Example Output

```
==================================================
RETAIL FORECAST PIPELINE - AUTOMATED GIT COMMIT & PUSH
==================================================
Repository: Retail-Forecast
Branch: main
Date: 2024-01-15 10:30:45

[INFO] Checking git configuration...
[SUCCESS] Git configuration OK

==================================================
ANALYZING REPOSITORY CHANGES
==================================================
[INFO] Found 5 changed files. Processing...
[INFO] Processing: logging_config.py (A)
[SUCCESS] Committed: logging_config.py
Commit #1: feat: add logging configuration module

[INFO] Processing: configs/training.yaml (M)
[SUCCESS] Committed: configs/training.yaml
Commit #2: fix: update ZenML training pipeline configuration

[INFO] Processing: materializers/prophet_materializer.py (M)
[SUCCESS] Committed: materializers/prophet_materializer.py
Commit #3: fix: resolve Prophet model materializer issues

[INFO] Processing: steps/data_visualizer.py (M)
[SUCCESS] Committed: steps/data_visualizer.py
Commit #4: fix: resolve Content Security Policy violations

[INFO] Processing: CHANGES.md (A)
[SUCCESS] Committed: CHANGES.md
Commit #5: docs: add comprehensive change documentation

==================================================
PUSHING CHANGES TO GITHUB
==================================================
[INFO] Pushing 5 commits to GitHub...
[SUCCESS] Successfully pushed 5 commits to GitHub!

==================================================
COMMIT SUMMARY
==================================================
[SUCCESS] Total commits created: 5

[INFO] Recent commits:
abc1234 docs: add comprehensive change documentation
def5678 fix: resolve Content Security Policy violations
ghi9012 fix: resolve Prophet model materializer issues
jkl3456 fix: update ZenML training pipeline configuration
mno7890 feat: add logging configuration module

[INFO] Repository status:
[SUCCESS] Automated commit and push completed!
```

### Dry Run Example

```bash
./auto_commit_push.sh --dry-run
```

```
==================================================
DRY RUN - PREVIEW OF CHANGES
==================================================
[1] logging_config.py (A)
Commit message: feat: add logging configuration module

[2] configs/training.yaml (M)
Commit message: fix: update ZenML training pipeline configuration

[3] CHANGES.md (A)
Commit message: docs: add comprehensive change documentation

[INFO] Would create 3 commits for the above files
[WARNING] No actual commits made (dry run mode)
```

## Error Handling

The scripts include comprehensive error handling for common issues:

- **Git not configured**: Prompts to set user name and email
- **Not a git repository**: Instructions to initialize git
- **No changes detected**: Graceful handling with informative messages
- **Push failures**: Attempts upstream setup and provides troubleshooting steps
- **Authentication issues**: Clear error messages with resolution steps

## Security Notes

- Scripts do not store or transmit credentials
- All git operations use your existing authentication setup
- Dry run mode allows safe preview of changes
- Each commit is created individually for better version control

## Customization

You can customize the commit message generation by editing the `generate_commit_message` function in either script. The function maps file patterns to specific commit message templates.

## Troubleshooting

### Common Issues

1. **"git: command not found"**
   - Install git on your system

2. **"Permission denied (publickey)"**
   - Set up SSH key authentication with GitHub
   - Or use personal access token for HTTPS

3. **"Git user name not configured"**
   - Run: `git config --global user.name "Your Name"`
   - Run: `git config --global user.email "your.email@example.com"`

4. **"Push failed"**
   - Check repository permissions
   - Verify remote URL is correct
   - Ensure authentication is properly configured

### Getting Help

Run either script with `--help` to see usage information:

```bash
./auto_commit_push.sh --help
./auto_commit_push.py --help
```

## License

These scripts are part of the Retail-Forecast project and follow the same licensing terms.