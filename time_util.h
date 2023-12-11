/*
 * File provided by Charles Curtsinger for the Sudoku Solver lab
 * for the CSC-213 class at Grinnell College for Fall 2023
 */
#if !defined(UTIL_H)
#define UTIL_H

#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

/**
 * Get the time in milliseconds since UNIX epoch
 */
size_t time_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) == -1) {
    perror("gettimeofday");
    exit(2);
  }

  // Convert timeval values to milliseconds
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#endif