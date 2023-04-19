/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

// #include "core/enum_types.h"
#include <cstdio>
#include <fmt/core.h>
#include <string>

namespace Aperture {

enum class LogLevel : char { info, debug, detail };

class Logger {
 private:
  static int m_rank;
  static LogLevel m_level;
  static std::string m_log_file;
  static std::FILE* m_file;

 public:
  Logger() {}
  ~Logger();

  static void init(int rank, LogLevel level, std::string log_file = "");
  static void set_log_level(LogLevel level) { m_level = level; }

  static bool open_log_file();

  template <typename... Args>
  static void print(const std::string& str, Args&&... args) {
    if (m_rank == 0) {
      fmt::print(str, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void err(const std::string& str, Args&&... args) {
    if (m_rank == 0) {
      fmt::print(stderr, str, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void print_err(const std::string& str, Args&&... args) {
    if (m_rank == 0) {
      fmt::print(stderr, "Error: " + str + "\n", std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void print_err_all(const std::string& str, Args&&... args) {
    fmt::print(stderr, "Rank {} Error: " + str + "\n", m_rank, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void print_info(const std::string& str, Args&&... args) {
    if (m_rank == 0) {
      fmt::print(str + "\n", std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void print_info_all(const std::string& str, Args&&... args) {
    fmt::print("Rank {}: " + str + "\n", m_rank, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void print_detail(const std::string& str, Args&&... args) {
    if (m_rank == 0 && m_level >= LogLevel::detail) {
      fmt::print("Detail: " + str + "\n", std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void print_detail_all(const std::string& str, Args&&... args) {
    if (m_level >= LogLevel::detail) {
      fmt::print("Rank {} Detail: " + str + "\n", m_rank, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void print_debug(const std::string& str, Args&&... args) {
    if (m_rank == 0 && m_level >= LogLevel::debug) {
      fmt::print("Debug: " + str + "\n", std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void print_debug_all(const std::string& str, Args&&... args) {
    if (m_level >= LogLevel::debug) {
      fmt::print("Rank {} Debug: " + str + "\n", m_rank, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void log_info(const std::string& str, Args&&... args) {
    if (m_rank == 0) {
      if (m_file == nullptr)
        if (!open_log_file()) {
          fmt::print("File can't be opened!");
          return;
        }
      fmt::print(m_file, str + "\n", std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void log_detail(const std::string& str, Args&&... args) {
    if (m_rank == 0 && m_level > LogLevel::info) {
      if (m_file == nullptr)
        if (!open_log_file()) return;
      fmt::print(m_file, str + "\n", std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void log_debug(const std::string& str, Args&&... args) {
    if (m_rank == 0 && m_level >= LogLevel::debug) {
      if (m_file == nullptr)
        if (!open_log_file()) return;
      fmt::print(m_file, "Debug: " + str + "\n", std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  static void log_debug_all(const std::string& str, Args&&... args) {
    if (m_file == nullptr)
      if (!open_log_file()) return;
    if (m_level >= LogLevel::debug) {
      fmt::print(m_file, "Debug: " + str + "\n", std::forward<Args>(args)...);
    }
  }
};

}  // namespace Aperture
