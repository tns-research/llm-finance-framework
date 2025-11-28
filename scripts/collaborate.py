#!/usr/bin/env python3
"""
Collaboration workflow helper for LLM Finance Framework
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command and return success status"""
    print(f"\n🔧 {description}")
    print(f"   Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Success"            return True
        else:
            print(f"   ⚠️  Exit code: {result.returncode}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"   ❌ Command not found: {cmd[0]}")
        return False

def setup_collaboration():
    """Set up collaboration environment"""
    print("🤝 Setting up collaboration environment...")

    steps = [
        (["git", "remote", "-v"], "Checking git remotes"),
        (["git", "status"], "Checking repository status"),
        (["git", "branch", "-a"], "Checking available branches"),
    ]

    for cmd, desc in steps:
        run_command(cmd, desc, check=False)

    print("\n📋 Collaboration Setup Complete!")
    print("Next steps:")
    print("1. Fork the repository on GitHub")
    print("2. Add upstream remote: git remote add upstream <original-repo-url>")
    print("3. Create feature branch: git checkout -b feature/your-feature")
    print("4. Run: python scripts/collaborate.py sync")

def sync_with_upstream():
    """Sync with upstream repository"""
    print("🔄 Syncing with upstream...")

    # Check if upstream remote exists
    result = subprocess.run(["git", "remote", "get-url", "upstream"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Upstream remote not found!")
        print("   Add it with: git remote add upstream <original-repo-url>")
        return False

    commands = [
        (["git", "fetch", "upstream"], "Fetching upstream changes"),
        (["git", "status"], "Checking current status"),
        (["git", "log", "HEAD..upstream/main", "--oneline"],
         "Checking what we're behind"),
    ]

    for cmd, desc in commands:
        if not run_command(cmd, desc, check=False):
            if "log" in cmd:  # log command might fail if we're up to date
                print("   (This is normal if you're up to date)")

    print("\n💡 To merge upstream changes:")
    print("   git rebase upstream/main  # or git merge upstream/main")
    return True

def create_feature_branch():
    """Help create a feature branch"""
    print("🌿 Creating feature branch...")

    # Ask for branch type and name
    print("\nBranch types:")
    print("1. feature/ - New functionality")
    print("2. research/ - Research experiments")
    print("3. fix/ - Bug fixes")
    print("4. docs/ - Documentation")

    branch_types = {
        "1": "feature",
        "2": "research",
        "3": "fix",
        "4": "docs"
    }

    try:
        choice = input("Choose branch type (1-4): ").strip()
        if choice not in branch_types:
            print("❌ Invalid choice")
            return False

        branch_type = branch_types[choice]
        name = input(f"Enter {branch_type} name (e.g., 'add-new-strategy'): ").strip()

        if not name:
            print("❌ Name cannot be empty")
            return False

        branch_name = f"{branch_type}/{name}"

        # Create and switch to branch
        commands = [
            (["git", "checkout", "-b", branch_name], f"Creating branch '{branch_name}'"),
            (["git", "status"], "Verifying branch creation"),
        ]

        success = True
        for cmd, desc in commands:
            if not run_command(cmd, desc):
                success = False

        if success:
            print(f"\n🎉 Branch '{branch_name}' created successfully!")
            print("Start working on your changes...")

        return success

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
        return False

def prepare_pull_request():
    """Prepare for pull request submission"""
    print("📋 Preparing for pull request...")

    commands = [
        (["git", "status"], "Checking current status"),
        (["git", "diff", "--name-only"], "Checking changed files"),
        (["python", "scripts/dev-workflow.py", "check"], "Running quality checks"),
    ]

    all_passed = True
    for cmd, desc in commands:
        if not run_command(cmd, desc, check=False):
            if "check" in cmd:
                print("   ⚠️  Fix quality issues before submitting PR")
                all_passed = False

    if all_passed:
        print("\n✅ Ready for pull request!")
        print("Next steps:")
        print("1. Commit your changes: git add . && git commit -m 'feat: Your changes'")
        print("2. Push to your fork: git push origin your-branch-name")
        print("3. Create pull request on GitHub")
        print("4. Fill out the PR template completely")

    return all_passed

def show_collaboration_status():
    """Show current collaboration status"""
    print("📊 Collaboration Status")

    commands = [
        (["git", "remote", "-v"], "Git remotes"),
        (["git", "branch", "-v"], "Local branches"),
        (["git", "status", "--short"], "Working directory status"),
        (["git", "log", "--oneline", "-5"], "Recent commits"),
    ]

    for cmd, desc in commands:
        print(f"\n{desc}:")
        run_command(cmd, check=False)

def main():
    actions = {
        "setup": setup_collaboration,
        "sync": sync_with_upstream,
        "branch": create_feature_branch,
        "pr": prepare_pull_request,
        "status": show_collaboration_status,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        print("🤝 LLM Finance Framework - Collaboration Helper")
        print()
        print("Available actions:")
        for action in actions:
            print(f"  {action:8} - {actions[action].__doc__ or action.replace('_', ' ')}")
        print()
        print("Examples:")
        print("  python scripts/collaborate.py setup   # Initial collaboration setup")
        print("  python scripts/collaborate.py sync    # Sync with upstream")
        print("  python scripts/collaborate.py branch  # Create feature branch")
        print("  python scripts/collaborate.py pr      # Prepare for PR")
        print("  python scripts/collaborate.py status  # Show collaboration status")
        return

    action = sys.argv[1]
    actions[action]()

if __name__ == '__main__':
    main()
