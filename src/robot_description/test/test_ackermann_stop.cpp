#include "../src/wheel_brake.hpp"
#include <cmath>
#include <iostream>

int main()
{
  const double gain = .0155;
  const double loaded = piracer::StopTorque(1.1975, gain, 10000.0);
  const double unloaded = piracer::StopTorque(0.0, gain, 10000.0);
  if (!(loaded < 0 && unloaded == 0)) return 1;
  if (std::abs(loaded + gain * 1.1975) > 1e-12) return 2;
  if (piracer::StopTorque(-1.0, gain, 10000.0) <= 0) return 3;
  if (piracer::StopTorque(100.0, gain, .01) != -.01) return 4;
  if (piracer::StopTorque(-100.0, gain, .01) != .01) return 5;
  if (piracer::StopTorque(1.0, 0.0, 10000.0) != 0) return 6;
  std::cout << "PASS: independent braking sign, rest, gain and effort limits\n";
}
