# Independent Codex Pull Request Review

Codex review must run as a new task with fresh context. The task that authored
the implementation must not be the sole reviewer of its own work.

## Requesting a Review

After validation jobs finish, comment on the pull request:

```text
@codex review
```

Request the review again after substantive updates. Until automatic review is
enabled in the repository's Codex settings, this explicit request is the
supported fallback.

## Required Reviewer Context

The reviewer must independently inspect:

- the linked issue, its acceptance criteria, and owner comments;
- the complete pull-request diff and directly affected code;
- repository documentation and established conventions;
- validation results and any unresolved prior review comments.

If no issue is linked, report that issue-compliance review is incomplete.

## Required Review Checks

Review for:

- omitted acceptance criteria, undocumented behavior, and unrelated changes;
- weak tests that execute code without proving behavior;
- assumptions of one human, two players, a fixed color, or a fixed list index;
- failures at any supported player count or map configuration;
- mutation during evaluation or GUI rendering;
- nondeterministic save/replay behavior or stale state-derived results;
- piece-conservation errors or bypasses around legal-action validation;
- globally applied map or optional-module rules;
- scoring, tie-breaking, displacement, and compound-action regressions;
- hidden-information leaks or AI inference attached to the wrong game state.

The reviewer should try to find defects, not defend the implementation.

## Findings Format

Use these severities:

- **Blocking:** rule violations, state corruption, security issues, broken
  compatibility, failed acceptance criteria, or missing critical tests.
- **Major:** incorrect multiplayer behavior, significant unhandled cases,
  frozen gameplay, stale AI results, or architecture that blocks planned work.
- **Minor:** maintainability issues, weak diagnostics, or missing noncritical
  tests.
- **Suggestion:** optional simplifications or out-of-scope improvements.

Post concise inline comments where useful and a summary containing:

```text
Issue compliance: Complete | Partial | Incomplete
Required checks: Passed | Failed
Blocking findings: N
Major findings: N
Minor findings: N
Unrelated changes: None | description
```

Avoid duplicate comments. On later reviews, mark prior findings resolved or do
not repeat them. Review is advisory at initial rollout; it never merges a pull
request. Human approval remains required.
