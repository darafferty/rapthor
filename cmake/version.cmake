# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# Based on the following blog post:
#   Compile-time version strings in CMake
#   Matt Keeter
#   https://www.mattkeeter.com/blog/2018-01-06-versioning/

set(GIT_COMMAND "git --git-dir ${CMAKE_CURRENT_SOURCE_DIR}/.git")

execute_process(COMMAND bash -c "${GIT_COMMAND} log --pretty=format:'%H' -n 1"
        OUTPUT_VARIABLE GIT_REV
        ERROR_QUIET)

# Check whether we got any revision (which isn't
# always the case, e.g. when someone downloaded a zip
# file from Github instead of a checkout)
if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "N/A")
    set(GIT_DIFF "")
    set(GIT_TAG "N/A")
    set(GIT_BRANCH "N/A")
else()
    execute_process(
        COMMAND bash -c "${GIT_COMMAND} diff --quiet --exit-code && echo +"
        OUTPUT_VARIABLE GIT_DIFF)
    execute_process(
        # No need to do an --exact-match, most recent tag should be OK
        COMMAND bash -c "${GIT_COMMAND} describe --tags"
        OUTPUT_VARIABLE GIT_TAG ERROR_QUIET)
    execute_process(
        COMMAND bash -c "${GIT_COMMAND} rev-parse --abbrev-ref HEAD"
        OUTPUT_VARIABLE GIT_BRANCH)

    string(STRIP "${GIT_REV}" GIT_REV)
    string(SUBSTRING "${GIT_REV}" 0 8 GIT_REV)
    string(STRIP "${GIT_DIFF}" GIT_DIFF)
    string(STRIP "${GIT_TAG}" GIT_TAG)
    string(STRIP "${GIT_BRANCH}" GIT_BRANCH)

    if(GIT_TAG MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+).*" OR GIT_TAG MATCHES "^([0-9]+\\.[0-9]+).*")
      # Regexp GIT_TAG on major.minor.match or major.minor, strip any commit info
      set(GIT_TAG "${CMAKE_MATCH_1}")
    elseif("${GIT_TAG}" STREQUAL "")
      message(WARNING "Could NOT find a matching (git) version tag. This is not an error, but only indicates that "
      "applications link against IDG might have troubles in retrieving VERSION info from IDG")
    else()
      message(FATAL_ERROR "Unrecognized (git) version tag ${GIT_TAG}")
    endif()
endif()

set(GIT_VERSION "const char* GIT_REV=\"${GIT_REV}${GIT_DIFF}\";
const char* GIT_TAG=\"${GIT_TAG}\";
const char* GIT_BRANCH=\"${GIT_BRANCH}\";\n")

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${GIT_VERSION_HEADER})
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/${GIT_VERSION_HEADER} GIT_VERSION_)
else()
    set(GIT_VERSION_ "")
endif()

if (NOT "${GIT_VERSION}" STREQUAL "${GIT_VERSION_}")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${GIT_VERSION_HEADER} "${GIT_VERSION}")
endif()
