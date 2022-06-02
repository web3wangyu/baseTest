#!/bin/sh
mkdir -p build
cmake -DCMAKE_BUILD_TYPE=Release -B build .
cmake --build build --verbose
