# Project instructions

- Prefer the smallest readable implementation that a student can explain.
- Keep future ROS 2 nodes single-purpose; use standard messages, simple callbacks/state, parameters and remapping.
- Do not add abstractions, frameworks, retries or guards without a demonstrated need.
- Keep simulation observations separate from physical PiRacer/OAK-D validation.
- Preserve asset provenance and keep the PiRacer MIT license separate from the suitcase CC BY 4.0 license.
- This repository contains the robot description, suitcase asset, baseline `empty_sky` scene, prototype `room15x15` scene and `piracer_sim.launch.py`. Additional scenes, controllers or perception behavior need their own approved task scope.

## ROS 2 commits

After a task under `src/` is complete and its required deterministic checks
pass, create a local commit for the verified task. Commit only task-owned ROS 2
package files under `src/`; keep `verification/`, generated `build/`,
`install/`, `log/`, caches and unrelated work out of code commits. The initial
repository commit may also include the root `.gitignore`, `AGENTS.md`,
`README.md` and `MIGRATION_MANIFEST.md` that establish this workspace.

Stage only task-owned paths and hunks. Do not use `git add -A`. Immediately
before committing, inspect `git diff --cached --name-only` and
`git diff --cached`; commit only when every staged path and hunk belongs to the
verified task. If unrelated changes are staged, leave the index untouched and
do not commit. If ownership of mixed file changes is unclear, leave the file
out and report the limitation.

Use a clear, natural action title in the repository's English Conventional
Commit style, plus a concise body describing the change and checks actually
run. Do not push unless the user asks.
