# Opis modelu PiRacer

Pakiet `robot_description` zawiera opis URDF/Xacro modelu, siatki wizualne i
teksty licencji. Nie uruchamia własnych węzłów. Pakiet `robot_simulation`
składa symulację Gazebo, rozwija Xacro i spawnuje model.

## Zawartość modelu

- `urdf/piracer.urdf.xacro` opisuje podwozie PiRacera, koła, skręt i ramy
  sensorów.
- `meshes/` zawiera siedem upstreamowych siatek STL: `board`, `chassis`,
  `cover`, `disk`, `motor`, `rpi` i `tire`.
- `third_party_licenses/` zawiera oryginalny tekst licencji użytych zasobów.
- Opcjonalna kamera RGB-D jest przybliżeniem renderowanym przez Gazebo. Nie
  symuluje pomiaru stereo OAK-D ani fizycznego montażu kamery. Xacro domyślnie
  jej nie dołącza; główny launch ustawia `camera_enabled:=true` domyślnie.
  Obrazy są publikowane w przestrzeni `/sim_camera`.

Pojazd i siedem siatek pochodzą z
[`PiRacer-Gazebo-Simulation`](https://github.com/lalywr2000/PiRacer-Gazebo-Simulation)
autorstwa Sujong Ha, z licencją MIT i przypiętym commitem
`db9f49c85e4417ec3566d0d87702f49febbd3ee1`. Lokalny tekst licencji znajduje
się w `third_party_licenses/PiRacer-Gazebo-Simulation-MIT.txt`.

## Napęd w bieżącym Xacro

Bieżący plik modelu używa upstreamowego pluginu ROS Humble
`gazebo_ros_ackermann_drive` z pakietu `gazebo_plugins`. Plugin steruje
czterema kołami jezdnymi i dwoma przegubami skrętu. Przyjmuje komendę
`/piracer/ackermann_cmd_vel`; launch symulacji uruchamia
`ackermann_twist_adapter`, który przekształca standardową komendę
`/cmd_vel` na format Ackermanna i publikuje wynik na
`/piracer/ackermann_cmd_vel`.

Ustawienia Xacro ograniczają prędkość do `0.20 m/s`, a kąt zadanego skrętu
pluginu do `0.554 rad`. Mechaniczne granice przegubów skrętu wynoszą osobno
`+/-0.785398 rad` (`+/-45°`). Wartości PID są wstępne i nie zostały
zweryfikowane w runtime. Plugin Ackermanna nie publikuje odometrii ani TF;
odometryczna ścieżka symulacji jest konfigurowana osobno przez
`robot_simulation`.

Suma jawnie przypisanych mas w rozwiniętym bieżącym URDF wynosi `1.018508 kg`.
To wartość wynikająca z parametrów modelu, a nie pomiar masy fizycznego
PiRacera.

Plugin napędu bazuje na `gazebo_ros_ackermann_drive` 3.9.0 (Apache-2.0).
Lokalna biblioteka `libpiracer_ackermann_drive.so` zachowuje jazdę i skręt
upstream. Przy zerowej komendzie hamuje każde tylne koło według jego własnej
prędkości, aby odciążenie jednego koła nie wyłączało hamowania drugiego.
Źródło, licencja i zakres zmiany: `third_party_licenses/gazebo_ackermann/`.

## Uruchomienie i zakres dowodów

Domyślne polecenie uruchamiające model to:

```bash
ros2 launch robot_simulation piracer_sim_base.launch.py
```

Launch wybiera `sala_laboratoryjna_production.world` i domyślnie spawnuje
pojazd na wysokości `z=0.019 m`. Szczegóły kompozycji symulacji należą do
README pakietu `robot_simulation`.

Raporty RUN48 i RUN49 dokumentują wcześniejsze, zapisane konfiguracje. Nie
potwierdzają bieżącego Xacro z pluginem Ackermanna ani jego parametrów.
Przytaczając te wyniki, należy podawać dokładną konfigurację z odpowiedniego
raportu. Żaden z tych raportów nie kwalifikuje fizycznego pojazdu.
