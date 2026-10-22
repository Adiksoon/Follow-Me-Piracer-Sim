# Production Gazebo scene and suitcase

Status: author-approved production default, recorded 2026-10-11.

## Canonical selection

- Production world: `worlds/sala_laboratoryjna_production.world`
- Default from `sim launch`: this production world. Use the launch argument
  `world:=...` only when a different scene is explicitly intended.
- Target entity: `suitcase_target`
- Gazebo model: `model://hard_trolley_red/meshes/suitcase.dae`
- Model package path: `simulation_assets/assets/hard_trolley_red/`
- Reference frame selected by the author:
  `verification/new_suitcase_preview_20261008/yolo_frames/production_scene_first_frame_20261009T122313Z.png`
- Reference image SHA-256:
  `13cf36ce67893807922e89bcc9c88f616779ac5b975bc8e2e48922fe8ad44ef7`
- Production world SHA-256:
  `71111ed9cfd933117c3ac7013bed41c85d342eb5cc6b8e4e0eddbeeb5ae39c7e`
- Suitcase DAE SHA-256:
  `e4d75b678e566289f15d58fdc7216c93f2199a34bb82377bd785d1e94a14f1ca`
- Furniture model hashes:
  - `lab_table_beige/model.sdf`:
    `f72bd4d9b5f0b50f6375589080842762b38dc8fccef6cca5c54cf7bb5e5a44b9`
  - `wooden_chair_blue_gray/model.sdf`:
    `6c0d4d2219af49b9b5737e0a25b36ebe5b69349640d4509eff1f21fe3027b28f`

The image named `01_klatka_kamery_scenariusz_testowy.png`, supplied for author
review, has the same SHA-256 as the reference frame above. It therefore binds
the production world and target model to one exact saved image instead of a
similar-looking scene or suitcase.

## Scene contents and boundaries

The selected world contains the lab tables, blue-gray chairs, room structure,
`suitcase_target`, and PiRacer spawn area. Its target visual mesh is the
`hard_trolley_red` DAE. The existing collision primitives and planar-move
plugin were copied unchanged from the author-reviewed world. The required
Gazebo models are installed under `simulation_assets/assets/` and resolved by
the launch file's existing `GAZEBO_MODEL_PATH`.

The legacy comparison world `worlds/sala_laboratoryjna.world` was removed from
the source package on 2026-10-22. Historical references to it remain in
`verification/` records; those records describe past runs and do not select a
runtime world. The removed world used a different `sim012_final_suitcase`
asset. The temporary textured suitcase probes under
`verification/suitcase_realistic_side_probe_*` are not selected or referenced
by this production scene.

## Suitcase attribution

The retained source archive is
`verification/new_suitcase_preview_20261008/source/hard_trolley_gazebo11.zip`
(SHA-256
`73e08aa6ae4c7f7857db4b04282833e73b5c80288271e65b6d94772bc9890602`). Its
included `CREDITS.txt` identifies the base model as “Hard Trolley Bag With
Adjustable Handle” by Adrian Saputra, Sketchfab model UID
`1758ad5207d943d597db8ff2b21bf827`, CC BY 4.0. The chosen variant recolors the
shell red; attribution and the source archive are retained with the project.

## Verification boundary

The production selection is tied to the exact saved frame and the copied world
and asset hashes above. Source files and model references were checked
offline; Gazebo was not started as part of recording this selection. This
record does not claim a new runtime, physics, or YOLO-quality measurement.
