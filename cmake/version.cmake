# Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Based on the following blog post:
#   Compile-time version strings in CMake
#   Matt Keeter
#   https://www.mattkeeter.com/blog/2018-01-06-versioning/

set(GIT_COMMAND "git --git-dir ${CMAKE_CURRENT_SOURCE_DIR}/.git")

execute_process(
  COMMAND bash -c "${GIT_COMMAND} log --pretty=format:'%H' -n 1"
  OUTPUT_VARIABLE GIT_REV
  ERROR_QUIET)

# Check whether we got any revision (which isn't
# always the case, e.g. when someone downloaded a zip
# file from Github instead of a checkout)
if("${GIT_REV}" STREQUAL "")
  set(GIT_TAG "")
  set(GIT_BRANCH "")
else()
  # Update GIT_REV.
  string(STRIP "${GIT_REV}" GIT_REV)
  string(SUBSTRING "${GIT_REV}" 0 8 GIT_REV)
  # Add '-dirty' if there are local changes.
  execute_process(COMMAND bash -c "${GIT_COMMAND} diff --quiet"
                  RESULT_VARIABLE GIT_DIFF_RESULT)
  if(GIT_DIFF_RESULT)
    string(APPEND GIT_REV "-dirty")
  endif()

  # Set GIT_TAG.
  execute_process(
    # No need to do an --exact-match, most recent tag should be OK
    COMMAND bash -c "${GIT_COMMAND} describe --tags"
    OUTPUT_VARIABLE GIT_TAG
    ERROR_QUIET)
  string(STRIP "${GIT_TAG}" GIT_TAG)

  # Regexp GIT_TAG on major.minor.match or major.minor, strip any commit info
  # Do not use 'OR' between the match conditions, since CMAKE_MATCH_1 becomes the last match.
  if(GIT_TAG MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+).*")
    set(GIT_TAG "${CMAKE_MATCH_1}")
  elseif(GIT_TAG MATCHES "^([0-9]+\\.[0-9]+).*")
    set(GIT_TAG "${CMAKE_MATCH_1}")
  elseif("${GIT_TAG}" STREQUAL "")
    message(
      WARNING
        "Could NOT find a matching (git) version tag. This is not an error, but only indicates that "
        "there may be a mismatch between the git tag and the IDG version.")
  else()
    message(FATAL_ERROR "Unrecognized (git) version tag ${GIT_TAG}")
  endif()

  # Set GIT_BRANCH.
  execute_process(COMMAND bash -c "${GIT_COMMAND} rev-parse --abbrev-ref HEAD"
                  OUTPUT_VARIABLE GIT_BRANCH)
  string(STRIP "${GIT_BRANCH}" GIT_BRANCH)
  if(${GIT_BRANCH} STREQUAL "HEAD")
    # HEAD is detached / does not point to a branch.
    set(GIT_BRANCH "")
  endif()
endif()
