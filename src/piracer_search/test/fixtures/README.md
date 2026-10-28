# SEARCH regression fixtures

These are project-owned test inputs preserved from the current branch workspace.
Tests require no experiment directories or rosbag files.

- `search_goal02_v5_cases.json`: byte-for-byte copy. Source: `experiments/search_offline_goal02/fixtures/search_goal02_v5_cases.json`; source SHA-256: `d5a9f1905cd30cf6ecd439eaa61a1347c952b428314c99b9581c4c56ef7759e1`.
- `search_goal02_v5_expected.json`: byte-for-byte copy. Source: `experiments/search_offline_goal02/fixtures/search_goal02_v5_expected.json`; source SHA-256: `00fa22682beaa169ef7efa9d0c3bd12246e017510a99a22a28906558af4fb9ee`.
- `cp40_goal2_footprint.json`: only the existing test input fields; values unchanged. Source: `verification/CP-SEARCH-42/CP40_GOAL2_FOOTPRINT.json`; source SHA-256: `674e6a81bfd71716ee3b77ab9d3fe8122371f86c25b7a59c07792680e0e677c6`.

The Goal02 oracle is hand-authored, as declared in the original fixture. The CP40 footprint data were extracted previously from the CP40 bag; the input record names bag SHA-256 `cc830e661f3ceef9b92548c646be3ceef9498f7a6f7b7253a90449906cb6b605`. No new simulation or measurements were performed.
