# Manifest migracji

Dla każdego przeniesionego pliku zapisano hash SHA-256 źródła i celu. Xacro ma zmieniony wyłącznie komentarz o lokalizacji informacji o upstreamie; pozostałe pliki z allowlisty zachowują identyczne bajty.

| Źródło w starym repo | Cel w tym repo | SHA-256 źródła | SHA-256 celu |
|---|---|---|---|
| ros2_ws/src/robot_description/urdf/piracer.urdf.xacro | src/robot_description/urdf/piracer.urdf.xacro | 5c90caec046193669154d52c6b738f9fc6dc142c4edd936986359d05577d5cfd | 118726b93b81f8ad20b30efab2c8d950a86734085cd62cd4ead4645d2583d593 |
| ros2_ws/src/robot_description/meshes/board.stl | src/robot_description/meshes/board.stl | 14f43e67cce0b5b644399f0aedb718f05818a7935e4d8a70fec7dc72f63d76d5 | 14f43e67cce0b5b644399f0aedb718f05818a7935e4d8a70fec7dc72f63d76d5 |
| ros2_ws/src/robot_description/meshes/chassis.stl | src/robot_description/meshes/chassis.stl | ec0f7ae46b29d1f44b74687146ef1e56942ebd40ed5e62f2050c81daa6bf76e7 | ec0f7ae46b29d1f44b74687146ef1e56942ebd40ed5e62f2050c81daa6bf76e7 |
| ros2_ws/src/robot_description/meshes/cover.stl | src/robot_description/meshes/cover.stl | 1cb6afd61161f27dea626b2d602dfa39c4e283cd62d985c361adafc0c5d265c7 | 1cb6afd61161f27dea626b2d602dfa39c4e283cd62d985c361adafc0c5d265c7 |
| ros2_ws/src/robot_description/meshes/disk.stl | src/robot_description/meshes/disk.stl | e353419b1248b58b18a46f72e17723899f4296c2a3886f84c7b1db5c83a59f31 | e353419b1248b58b18a46f72e17723899f4296c2a3886f84c7b1db5c83a59f31 |
| ros2_ws/src/robot_description/meshes/motor.stl | src/robot_description/meshes/motor.stl | edcfef888f68144b0b26376e8e86249d7e15f549e1e0d9aa790a2878050d2b73 | edcfef888f68144b0b26376e8e86249d7e15f549e1e0d9aa790a2878050d2b73 |
| ros2_ws/src/robot_description/meshes/rpi.stl | src/robot_description/meshes/rpi.stl | e15ed13c810f9ab051673fc2755cc046e8ff3e967b1d96af4b2f252d6969948f | e15ed13c810f9ab051673fc2755cc046e8ff3e967b1d96af4b2f252d6969948f |
| ros2_ws/src/robot_description/meshes/tire.stl | src/robot_description/meshes/tire.stl | 3180405d3a70e60045afb68588d1e6b705be29625d299c02afcbb1cebaf03373 | 3180405d3a70e60045afb68588d1e6b705be29625d299c02afcbb1cebaf03373 |
| ros2_ws/src/robot_description/third_party_licenses/PiRacer-Gazebo-Simulation-MIT.txt | src/robot_description/third_party_licenses/PiRacer-Gazebo-Simulation-MIT.txt | eadb9b0371cfa3fcf647dacfb45cea768ffafc1e75da8fb9600871b6934c7d66 | eadb9b0371cfa3fcf647dacfb45cea768ffafc1e75da8fb9600871b6934c7d66 |
| ros2_ws/src/robot_simulation/assets/sim012_final_suitcase/source_Hard-Trolley-Bag-With-Adjustable-Handle.dae | src/simulation_assets/assets/sim012_final_suitcase/source/source_Hard-Trolley-Bag-With-Adjustable-Handle.dae | f98c32dc6c2f2df2b5317fc76c0d2ae0ffc0d05bff7d79125d6ac3ff6284771c | f98c32dc6c2f2df2b5317fc76c0d2ae0ffc0d05bff7d79125d6ac3ff6284771c |
| ros2_ws/src/robot_simulation/assets/sim012_final_suitcase/suitcase.obj | src/simulation_assets/assets/sim012_final_suitcase/runtime/suitcase.obj | 20620abe87423fc797f19e10550cebf3feeeadd24b9053b000089c1b269544c3 | 20620abe87423fc797f19e10550cebf3feeeadd24b9053b000089c1b269544c3 |
| ros2_ws/src/robot_simulation/assets/sim012_final_suitcase/suitcase-sketchfab-gazebo.mtl | src/simulation_assets/assets/sim012_final_suitcase/runtime/suitcase-sketchfab-gazebo.mtl | 634a3ef653b2c0d7e76f3c6d7ec041ee6b9e001ac25b7309d03b924df9ced0a0 | 634a3ef653b2c0d7e76f3c6d7ec041ee6b9e001ac25b7309d03b924df9ced0a0 |

## Later Xacro changes

The hashes above preserve the original migration-time source and destination.
The destination Xacro was later updated with the simulation RGB-D prototype
profile; its current SHA-256 is
`04af89583f2407ab06fd54b85f2f8202f05e423831bfe9d170d88be16ede8c0e`.
