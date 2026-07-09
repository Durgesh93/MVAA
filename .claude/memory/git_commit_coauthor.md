---
name: git-commit-coauthor
description: User wants Claude listed as co-author on commits pushed on their behalf
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b303adc7-e9c9-4abb-9f35-e5cb82c2dee8
---

When creating git commits for code changes made on the user's behalf (this applies across all their projects, not just MVAA), always include a `Co-Authored-By: Claude <noreply@anthropic.com>` trailer in the commit message.

**Why**: user explicitly asked for this (2026-07-08) after a session where several substantial code changes (MVAA `supervised`/`focal-pseudo-quality` branches) were made via file edits but not yet committed — surfaced the general expectation that pushed code should carry Claude co-authorship.

**How to apply**: this already matches the default Claude Code git-commit instructions, but treat it as a firm, explicit preference for this user rather than just the tool default — don't skip it even in contexts where it might otherwise seem optional. Only create commits when the user actually asks for a commit (don't commit proactively/silently).
