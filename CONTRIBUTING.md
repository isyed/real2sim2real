# Contributing

Keep related changes on one feature branch. Open or update its pull request when
syncing with the remote repository. Run the checks in [TESTING](docs/TESTING.md)
and keep private captures, credentials, weights and generated assets outside Git.

`main` requires a pull request, approval from the sole code owner `@isyed`, resolved
review conversations, and passing CPU checks on Python 3.11 and 3.12. New commits
dismiss stale approvals. The rule applies to administrators, with no review bypass;
force pushes and branch deletion are disabled. `.github/CODEOWNERS` assigns every
path, including itself and workflows, to the owner. The matching API configuration
is in `.github/main-protection.json`; enforcement lives in GitHub repository
settings, not in that JSON file alone.

GitHub does not allow authors to approve their own pull requests. Automated work
must therefore open PRs using a separate authorized contributor or GitHub App
identity so `@isyed` can review them. Do not approve on the owner's behalf, weaken
protection, or impersonate another author to merge a blocked PR. Leave a concrete
reviewable PR and report the missing review. The repository owner can still edit
repository settings; these rules govern merges while enabled.

Delete a feature branch only after its commits are reachable from `main`. Preserve
unmerged work, other worktrees and failed experiment evidence.
