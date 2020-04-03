#include "utils/timer.h"
#include "utils/logger.h"

using namespace Aperture;
using namespace std::chrono;

std::unordered_map<std::string, high_resolution_clock::time_point>
    timer::t_stamps;
high_resolution_clock::time_point timer::t_now =
    high_resolution_clock::now();
int stamp_depth = 0;

void
timer::stamp(const std::string& name) {
  t_stamps[name] = high_resolution_clock::now();
  stamp_depth += 1;
}

void
timer::show_duration_since_stamp(const std::string& routine_name,
                                 const std::string& unit,
                                 const std::string& stamp_name) {
  t_now = high_resolution_clock::now();
  for (int i = 0; i < stamp_depth + 1; i++) {
    Logger::print("---");
  }
  std::string elapsed_time = "";
  if (unit == "second" || unit == "s") {
    auto dur = duration_cast<duration<float, std::ratio<1, 1>>>(
        t_now - t_stamps[stamp_name]);
    elapsed_time = fmt::format("{}s", dur.count());
  } else if (unit == "millisecond" || unit == "ms") {
    auto dur =
        duration_cast<milliseconds>(t_now - t_stamps[stamp_name]);
    elapsed_time = fmt::format("{}ms", dur.count());
  } else if (unit == "microsecond" || unit == "us") {
    auto dur =
        duration_cast<microseconds>(t_now - t_stamps[stamp_name]);
    elapsed_time = fmt::format("{}µs", dur.count());
  } else if (unit == "nanosecond" || unit == "ns") {
    auto dur = duration_cast<nanoseconds>(t_now - t_stamps[stamp_name]);
    elapsed_time = fmt::format("{}ns", dur.count());
  }
  if (routine_name == "" && stamp_name == "") {
    Logger::print_info(" Time for default clock is {}", elapsed_time);
  } else if (routine_name == "") {
    Logger::print_info(" Time for {} is {}", stamp_name, elapsed_time);
  } else {
    Logger::print_info(" Time for {} is {}", routine_name, elapsed_time);
  }
  stamp_depth -= 1;
}

float
timer::get_duration_since_stamp(const std::string& unit,
                                const std::string& stamp_name) {
  stamp_depth -= 1;
  t_now = high_resolution_clock::now();
  if (unit == "millisecond" || unit == "ms") {
    auto dur =
        duration_cast<milliseconds>(t_now - t_stamps[stamp_name]);
    return dur.count();
  } else if (unit == "microsecond" || unit == "us") {
    auto dur =
        duration_cast<microseconds>(t_now - t_stamps[stamp_name]);
    return dur.count();
  } else if (unit == "nanosecond" || unit == "ns") {
    auto dur = duration_cast<nanoseconds>(t_now - t_stamps[stamp_name]);
    return dur.count();
  } else {
    auto dur = duration_cast<duration<float, std::ratio<1, 1>>>(
        t_now - t_stamps[stamp_name]);
    return dur.count();
  }
}
