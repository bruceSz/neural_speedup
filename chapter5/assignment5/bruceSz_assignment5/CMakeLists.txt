#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

project(sampleOnnxBert)
cmake_minimum_required(VERSION 3.15)

get_filename_component(SAMPLES_DIR ../ ABSOLUTE)
include_directories(${TARGET_NAME}
    /usr/local/cuda/include
    /data/peihy/envs/TensorRT-8.4.0.6/include
    ${SAMPLES_DIR}/common
)

link_directories(
    /data/peihy/envs/TensorRT-8.4.0.6/lib/
    /usr/local/cuda/lib64
)


set(SAMPLES_COMMON_SOURCES
    ${SAMPLES_DIR}/common/logger.cpp
)





add_executable(sampleonnxbert
    sampleOnnxBert.cpp
    ${SAMPLES_COMMON_SOURCES}
)

set(TARGET_NAME sampleonnxbert)
target_link_libraries( ${TARGET_NAME}
    nvinfer
    nvonnxparser
    cudart
)

#include(../CMakeSamplesTemplate.txt)
