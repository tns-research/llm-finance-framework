#!/usr/bin/env python3
"""
Development workflow helper for LLM Finance Framework
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔧 {description}")
    print(f"   Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def setup_dev_environment():
    """Set up development environment"""
    print("🚀 Setting up development environment...")

    # Install development dependencies
    if not run_command([
        sys.executable, "-m", "pip", "install",
        "-e", ".[dev]"
    ], "Installing development dependencies"):
        return False

    print("\n✅ Development environment ready!")
    return True

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    return run_command([
        "python", "-m", "pytest", "tests/",
        "-v", "--tb=short", "--cov=src/", "--cov-report=term-missing"
    ], "Running test suite with coverage")

def run_linting():
    """Run linting and formatting checks"""
    print("\n🔍 Running code quality checks...")

    checks = [
        (["flake8", "src/", "tests/", "--max-line-length=127", "--extend-ignore=E203,W503"],
         "Flake8 linting"),
        (["black", "--check", "--diff", "src/", "tests/"],
         "Black formatting check"),
        (["isort", "--check-only", "--diff", "src/", "tests/"],
         "Import sorting check"),
    ]

    all_passed = True
    for cmd, desc in checks:
        if not run_command(cmd, desc):
            all_passed = False

    return all_passed

def run_type_checking():
    """Run type checking"""
    print("\n📝 Running type checking...")
    return run_command([
        "mypy", "src/", "--ignore-missing-imports"
    ], "MyPy type checking")

def run_full_check():
    """Run full development check suite"""
    print("🔬 Running full development check suite...")

    checks = [
        ("setup", setup_dev_environment),
        ("lint", run_linting),
        ("types", run_type_checking),
        ("tests", run_tests),
    ]

    results = {}
    for name, func in checks:
        results[name] = func()

    print("\n📊 Check Results:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name.upper()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All checks passed! Ready for commit.")
    else:
        print("\n⚠️  Some checks failed. Please fix issues before committing.")

    return all_passed

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Development workflow helper')
    parser.add_argument('action', choices=[
        'setup', 'test', 'lint', 'types', 'check', 'version'
    ], help='Action to perform')

    args = parser.parse_args()

    if args.action == 'setup':
        setup_dev_environment()
    elif args.action == 'test':
        run_tests()
    elif args.action == 'lint':
        run_linting()
    elif args.action == 'types':
        run_type_checking()
    elif args.action == 'check':
        run_full_check()
    elif args.action == 'version':
        # Import and run version script
        sys.path.insert(0, str(Path(__file__).parent))
        import version
        version.main()

if __name__ == '__main__':
    main()
