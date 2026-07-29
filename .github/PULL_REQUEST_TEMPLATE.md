## Linked Issue

Closes #

## Summary

Describe what changed and why.

## Validation

- [ ] `python tools/validate_pr.py` passes
- [ ] Unit and integration tests pass
- [ ] Formatting passes
- [ ] Linting passes
- [ ] Type checks pass, where configured
- [ ] Manual smoke test completed, where appropriate

## Scope and Safety

- [ ] This PR contains only changes relevant to the linked issue
- [ ] All supported player counts were considered
- [ ] Save/replay compatibility and deterministic behavior were considered
- [ ] Piece conservation and legal-action boundaries were considered
- [ ] Optional maps/modules remain scoped to configurations that enable them
- [ ] No hidden player information is exposed

## Independent AI Review

- [ ] A fresh-context review was requested with `@codex review`
- [ ] The reviewer compared the complete diff with the linked issue
- [ ] Blocking findings are resolved
- [ ] Major findings are resolved or explicitly acknowledged

The implementation conversation's self-assessment does not count as the
independent review. Follow [.github/CODEX_REVIEW.md](CODEX_REVIEW.md).
