# PiRacer robot description

Current steering amendment (2026-10-18): the front-wheel limit is now 45 degrees
as an approximate author-observed simulation assumption. The matching planning
radius is 0.25 m. Historical RUN48/RUN49 results below apply to their recorded
settings and do not qualify the new tighter-turn behavior or physical hardware.

Pochodzenie: [PiRacer-Gazebo-Simulation](https://github.com/lalywr2000/PiRacer-Gazebo-Simulation), Sujong Ha (lalywr2000), MIT, przypięty upstream commit db9f49c85e4417ec3566d0d87702f49febbd3ee1. Oryginalny tekst licencji znajduje się w `third_party_licenses/PiRacer-Gazebo-Simulation-MIT.txt`.

Zachowano strukturę modelu Xacro i siedem upstreamowych meshy STL. Opis został dostosowany do pakietu `robot_description` i ramy `base_link`; kamera jest oznaczona jako przybliżenie symulacyjne, nie jako pomiar fizycznego montażu.

## Aktywny baseline produkcyjny symulacji

`urdf/piracer.urdf.xacro` jest aktywnym baseline'em symulacyjnym PiRacera o masie 1 kg. Używa `gazebo_ros_diff_drive` do napędu tylnych kół oraz `piracer_kinematic_steering` do ograniczonego, kinematycznego sterowania przednimi kołami. Domyślny launch `ros2 launch robot_simulation piracer_sim.launch.py` uruchamia scenę `sala_laboratoryjna_production.world` i spawnuje model na wysokości `z=0.019 m`.

Status oznacza zatwierdzony punkt startowy do dalszego rozwoju Follow-Me, YOLO i visual odometry. RUN48 sprawdził ograniczoną jazdę do przodu i tyłu, skręt w obie strony, zatrzymania, kontakt czterech opon oraz brak LCP i wartości niefinitywnych. RUN49 potwierdził produkcyjny launch, RGB-D, wyjście `/visual_odom` i zatrzymanie po krótkiej komendzie ruchu. Astra wydała `PASS` na użycie tego wariantu jako baseline'u dla dalszego rozwoju.

To zatwierdzenie dotyczy baseline'u **symulacyjnego**. Nie potwierdza dokładności visual odometry, jakości detekcji YOLO ani zachowania Follow-Me, a także nie kwalifikuje fizycznego PiRacera. Szczegóły i surowe wyniki pozostają w `verification/piracer_1kg_physics_stabilization_20261011/RUN48_diff_drive_steering_20261013/` oraz `verification/piracer_1kg_physics_stabilization_20261011/RUN49_production_integration_20261013/`.
