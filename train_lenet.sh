#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/DAN/alexnet/solver.prototxt $@
