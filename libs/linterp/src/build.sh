#!/bin/bash

name=linterp
target=$name.cpp
output=$name$(python3-config --extension-suffix)

clang++ \
  -O3 -Wall \
  -std=c++11 \
  $(python3 -m pybind11 --includes) \
  -shared -fPIC $target \
  -o $output \

  # -undefined dynamic_lookup \
