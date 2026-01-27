# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility shim: re-export everything from the actual module location
from autoparallel.graph_passes.auto_bucketing import *  # noqa: F401, F403
