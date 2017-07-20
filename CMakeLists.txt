cmake_minimum_required(VERSION 2.8)
project(idg)

set(IDG_OnePackage True)
add_subdirectory("idg-lib")
add_subdirectory("idg-bin")
add_subdirectory("idg-api")
