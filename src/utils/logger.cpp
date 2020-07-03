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

#include "utils/logger.h"

namespace Aperture {

int Logger::m_rank = 0;
LogLevel Logger::m_level = LogLevel::info;
std::string Logger::m_log_file = "";
std::FILE* Logger::m_file = nullptr;

void
Logger::init(int rank, LogLevel level, std::string log_file) {
  m_rank = rank;
  m_level = level;
  m_log_file = log_file;
}

Logger::~Logger() {
  if (m_file != nullptr) {
    fclose(m_file);
  }
}

bool
Logger::open_log_file() {
  m_file = std::fopen(m_log_file.c_str(), "w");
  if (!m_file) {
    print_err("Can't open log file, unable to log to file\n");
    return false;
  }
  return true;
}

}  // namespace Aperture
