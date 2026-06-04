---
name: 🐛 Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] Brief description of the issue"
labels: bug
assignees: ''
---

## 🐛 Bug Description
**What happened?**
A clear and concise description of what the bug is.

**What should have happened?**
What you expected to happen instead.

## 🔍 Steps to Reproduce
1. Go to '...'
2. Run command '....'
3. Configure with '....'
4. See error

## 💻 Environment
- **OS**: [e.g., Windows 10, macOS 12.0, Ubuntu 20.04]
- **Python Version**: [e.g., 3.8.10]
- **Framework Version**: [e.g., commit hash or version]
- **Dependencies**: [output of `pip list` or key packages]

## 📊 Error Messages
```
# If applicable, paste the full error traceback here
```

## 📁 Configuration Used
```python
# Paste relevant config.py settings or command used
LLM_PROVIDER = "dummy"  # or "openrouter" / "claude_code" / "claude_code_subagents"
DATA_SOURCE = "vendored"  # or "csv" / "stooq"
ACTIVE_EXPERIMENT = "baseline"  # or other
TEST_MODE = True  # or False
# ... other relevant settings
```

## 📈 Expected Behavior Context
- Were you running a real LLM experiment or dummy model?
- What experiment configuration were you using?
- Did this work before? When did it break?

## 📎 Additional Context
- Screenshots of error messages
- Log files (if available)
- Any other relevant information

## ✅ Verification Steps
- [ ] Bug can be reproduced with dummy model
- [ ] Bug occurs with specific experiment configuration
- [ ] Bug is not related to API key issues (if applicable)
