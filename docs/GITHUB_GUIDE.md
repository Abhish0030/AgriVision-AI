# GitHub upload guide

The audit found an initialized main branch with no commits and no remote. All
source files are untracked; they have not been staged or published. First inspect
your current state again, since it may change after this audit:

```powershell
git status
git remote -v
```

Create an empty repository named `AgriVision-AI` in your GitHub account, without a
separate generated README. Add origin only when `git remote -v` shows no origin:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/AgriVision-AI.git
```

Replace YOUR_USERNAME and review the resulting remote. Do not replace an existing
remote or force-push over its history. No remote or publication is assumed in this workspace.
Run these commands in VS Code's terminal at the project root:

```powershell
git status
git add .
git status --short
git diff --cached --stat
```

Review staged filenames. `.venv`, dataset/output contents, checkpoints, archives
and common credential files are ignored. Ignore rules do not protect files that
were already tracked. Keep datasets, weights, Drive copies and secrets out of the
commit; do not embed authentication tokens in notebook cells or remote URLs.

After review, replacing `YOUR_USERNAME`:

```powershell
git commit -m "Initial release: AgriVision AI"
git push origin main
```

If Git requests author details, set your own `git config user.name` and
`git config user.email`. Use the supported credential manager/browser sign-in.
If a remote already exists, inspect `git remote -v` before changing it.

Open the notebook using Colab's **File → Open notebook → GitHub** tab, or replace
the username in:

```text
https://colab.research.google.com/github/YOUR_USERNAME/AgriVision-AI/blob/main/notebooks/AgriVision_AI_Complete.ipynb
```

Set the notebook's REPOSITORY_URL to your repository. Push later changes from
VS Code. In a clean Colab checkout, `git pull --ff-only` updates source; restart
the session afterward to avoid stale imports. For reproducibility, record/check
out the intended commit. The notebook saves the cloned commit in Drive metadata.
