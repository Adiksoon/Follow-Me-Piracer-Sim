# Symulacja ROS 2

Workspace ROS 2 Humble z modelem PiRacera, assetem walizki oraz początkowym
uruchomieniem Gazebo Classic.

## Pakiety

- `robot_description` — Xacro/URDF PiRacera, siedem wymaganych meshy i licencja upstreamu.
- `simulation_assets` — źródłowy DAE oraz runtime OBJ/MTL walizki z atrybucją.
- `robot_simulation` — launch `piracer_sim.launch.py`, scena FPS `empty_sky.world`, scena `room15x15.world` oraz proste modele ławek i krzeseł.
- Opcjonalny YOLO korzysta z istniejącego pakietu `robot_perception` w głównym workspace `ros2_ws`.

## Budowanie i uruchamianie

W Ubuntu/WSL z ROS 2 Humble i Gazebo Classic:

    cd /home/zygar/dev_praca_inz/symulacja
    source /opt/ros/humble/setup.bash
    colcon build --base-paths src
    source install/setup.bash
    ros2 launch robot_simulation piracer_sim.launch.py

Domyślne uruchomienie otwiera Gazebo GUI, kamerę RGB-D z oknem podglądu obrazu
kolorowego, panel sterowania walizką i scenę `room15x15.world`. W oknie
`rqt_image_view` można wybrać także temat głębi. Lekki wariant do porównań FPS można uruchomić
bez GUI i sensora w pustym świecie:

    ros2 launch robot_simulation piracer_sim.launch.py gui:=false camera_enabled:=false suitcase_control_gui:=false world:=$(ros2 pkg prefix robot_simulation)/share/robot_simulation/worlds/empty_sky.world

## YOLO na symulowanej kamerze

Detektor jest opcjonalny i domyślnie wyłączony, żeby zwykły launch zachował
lekki profil. Uruchomienie YOLOX-M z podglądem obrazu z narysowanymi ramkami:

    source /opt/ros/humble/setup.bash
    source /home/zygar/dev_praca_inz/ros2_ws/install/setup.bash
    source /home/zygar/dev_praca_inz/symulacja/install/setup.bash
    ros2 launch robot_simulation piracer_sim.launch.py yolo_enabled:=true camera_view_topic:=/detections/debug_image

Gdy `yolo_enabled:=true`, launch używa domyślnie wag
`object_detection_yolox_m_20260918.onnx`, progu 0.35 i limitu 8 wątków OpenCV.
Inny plik można wskazać argumentem `yolo_model_path:=/pełna/ścieżka/model.onnx`.
Temat wejściowy, próg i limit wątków mają osobne argumenty launch. Tablice
detekcji są publikowane na `/detections`, a obraz z ramkami na
`/detections/debug_image`. Wybierz ten temat w `rqt_image_view`, by oglądać
wynik YOLO zamiast surowego RGB.

`room15x15.world` ma otwarte od góry wnętrze 15 x 15 m z czterema ścianami wysokości 3 m; nie ma
sufitu, lamp ani cieni, a neutralne światło ambientowe pozostaje tylko dla
widoczności obiektów. Ściany i podłoga mają jednolity, neutralny kolor.
Ławki i krzesła są ustawione w dwóch kolumnach szkolnych, skierowanych ku
północnej ścianie; pośrodku pozostaje szeroki przejazd. Domyślna pozycja
`base_link` przy uruchomieniu w tej scenie to `(0, 0, 0)`, a yaw wynosi `0`.
Meble są prototypową geometrią SDF do układu sceny i testów; nie są zatwierdzonym
finalnym assetem do pracy dyplomowej. Inną scenę można wskazać argumentem
`world:=/ścieżka/do/pliku.world`.

## Sterowanie walizką

Panel sterowania jest uruchamiany domyślnie; launch przekierowuje jego domyślny
temat `/cmd_vel` do `/sim012_target/cmd_vel`, subskrybowanego przez walizkę.
Pole `stamped` musi być odznaczone, aby wysyłać `Twist`. Panel zapamiętuje to
ustawienie przy zamknięciu. Suwakami ustawiaj prędkość liniową i obrót;
przycisk Stop zeruje prędkości. Do pomiarów FPS wyłącz panel argumentem
`suitcase_control_gui:=false`.

Walizka w `room15x15.world` korzysta z `gazebo_ros_planar_move`, tak jak w
głównym repozytorium. Alternatywnie można wysłać jej prędkość z terminala:

    ros2 topic pub -r 10 /sim012_target/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.25}, angular: {z: 0.0}}"

Zatrzymanie walizki:

    ros2 topic pub --once /sim012_target/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

Plugin publikuje też `/sim012_target/odom`. Domyślny `empty_sky.world` pozostaje
bez walizki i zachowuje rolę lekkiego baseline'u FPS.

## Kamera RGB-D

Model ma jeden syntetyczny sensor głębi Gazebo z pluginem
`libgazebo_ros_camera.so`; przy `camera_enabled:=true` publikuje obraz RGB,
głębię i CameraInfo. Bieżący prototyp `SYMULACJA_RGBD_PROTOTYPE_V1` ustawia
640 × 360, 30 Hz, HFOV 65,36° (VFOV około 39,7° z proporcji obrazu) oraz
clipping renderera 0,05–20 m. HFOV wynika z fabrycznej kalibracji RGB kamery
OAK-D Lite przy 3840 × 2160 i zachowuje jej proporcję 16:9 po zmniejszeniu
obrazu; 640 × 360 przy 30 Hz oznacza o 64% mniej pikseli na sekundę niż
poprzednie 800 × 800 przy 30 Hz. Liczby te opisują ustawienia i obciążenie
obliczeniowe, nie zmierzony wzrost FPS.

Jest to profil prowizoryczny: aktywna rozdzielczość, częstotliwość, korekcja
obrazu i sposób wyrównania RGB/depth fizycznej kamery nie zostały odczytane.
Gazebo generuje idealną głębię z geometrii sceny; nie odtwarza tu stereoskopii,
szumu ani kalibracji głębi OAK-D. Parametry są domyślnymi argumentami w
`src/robot_description/urdf/piracer.urdf.xacro`; dotychczasowe położenie kamery
pozostaje bez zmian. Uruchomienie pokoju z kamerą:

    ros2 launch robot_simulation piracer_sim.launch.py gui:=true camera_enabled:=true world:=$(ros2 pkg prefix robot_simulation)/share/robot_simulation/worlds/room15x15.world

Tematy obrazu to `/piracer/sim_camera/color/image_raw` i
`/piracer/sim_camera/depth/image_raw`; domyślne okno pokazuje pierwszy z nich,
a drugi można wybrać z listy tematów. CameraInfo jest dostępne pod
`/piracer/sim_camera/color/camera_info` i
`/piracer/sim_camera/depth/camera_info`.

Podgląd uruchom w drugim terminalu po załadowaniu środowiska ROS:

    source /opt/ros/humble/setup.bash
    source /home/zygar/dev_praca_inz/symulacja/install/setup.bash
    ros2 run rqt_image_view rqt_image_view

W selektorze obrazu wybierz jeden z tematów `color/image_raw` lub
`depth/image_raw` z przestrzeni `/piracer/sim_camera`.

## Pochodzenie i licencje

Kod/model PiRacera pochodzi z PiRacer-Gazebo-Simulation (MIT); szczegóły i lokalne modyfikacje opisuje `src/robot_description/README.md`. Walizka Sketchfab jest udostępniona na CC BY 4.0; atrybucja i modyfikacje są w `src/simulation_assets/README.md`. Licencje dotyczą różnych assetów i nie są jedną licencją całego repozytorium.
