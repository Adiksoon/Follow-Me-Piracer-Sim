# DATA-01 scene fixture

`data01_room.world` is the unmodified
`src/robot_simulation/worlds/sala_laboratoryjna_production.world` from project
commit `45bbe0c23f49cba32bc6ab227a0569e565b58d6c`.
Its SHA-256 is `a2db8fa83e785b87629c93dd750ec76e4744f44113989ed68e1eee70cdefe188`,
the existing DATA-01 hash enforced by `suitcase_dataset_generator.py`.

Dataset unit tests use this frozen input instead of the evolving SEARCH world.
The production capture guard and accepted hashes are unchanged. This file is
only a test fixture; it is not installed as a replacement production world.
