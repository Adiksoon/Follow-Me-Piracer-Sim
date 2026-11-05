#pragma once
#include <algorithm>

namespace piracer
{
inline double StopTorque(double wheel_rate, double p_gain, double effort_limit)
{
  return std::clamp(-p_gain * wheel_rate, -effort_limit, effort_limit);
}
}
