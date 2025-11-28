#!/usr/bin/env python3
"""
Version management script for LLM Finance Framework
"""

import re
import subprocess
from pathlib import Path

def get_current_version():
    """Get current version from pyproject.toml"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, 'r') as f:
        content = f.read()

    version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if version_match:
        return version_match.group(1)
    else:
        raise ValueError("Could not find version in pyproject.toml")

def update_version(new_version):
    """Update version in pyproject.toml"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, 'r') as f:
        content = f.read()

    # Update version
    updated_content = re.sub(
        r'version\s*=\s*["\']([^"\']+)["\']',
        f'version = "{new_version}"',
        content
    )

    with open(pyproject_path, 'w') as f:
        f.write(updated_content)

    print(f"Updated version to {new_version}")

def create_git_tag(version):
    """Create a git tag for the version"""
    try:
        # Create annotated tag
        subprocess.run(['git', 'tag', '-a', f'v{version}', '-m', f'Release version {version}'], check=True)
        print(f"Created git tag v{version}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create git tag: {e}")

def bump_version(bump_type):
    """Bump version according to semantic versioning"""
    current = get_current_version()
    major, minor, patch = map(int, current.split('.'))

    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        raise ValueError("bump_type must be 'major', 'minor', or 'patch'")

    new_version = f"{major}.{minor}.{patch}"
    update_version(new_version)
    return new_version

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Version management for LLM Finance Framework')
    parser.add_argument('action', choices=['get', 'set', 'bump', 'tag'],
                       help='Action to perform')
    parser.add_argument('--version', help='Version to set (for set action)')
    parser.add_argument('--bump', choices=['major', 'minor', 'patch'],
                       help='Version bump type (for bump action)')

    args = parser.parse_args()

    if args.action == 'get':
        print(get_current_version())
    elif args.action == 'set':
        if not args.version:
            parser.error("--version is required for set action")
        update_version(args.version)
    elif args.action == 'bump':
        if not args.bump:
            parser.error("--bump is required for bump action")
        new_version = bump_version(args.bump)
        print(f"Bumped version to {new_version}")
    elif args.action == 'tag':
        version = get_current_version()
        create_git_tag(version)

if __name__ == '__main__':
    main()
