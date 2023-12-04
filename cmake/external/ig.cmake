# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

project(dependency NONE)

include(ExternalProject)

# cmake-format: off
ExternalProject_Add(ig_lib
  URL               https://github.com/intel/xFasterTransformer/releases/download/IntrinsicGemm/ig_v1.1.tar.gz
  URL_HASH          MD5=47e5a2cd021caad2b1367c0b71dff2e7
  TIMEOUT           60
  SOURCE_DIR        ${CMAKE_SOURCE_DIR}/third_party/ig
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
# cmake-format: on

# add_library(ig STATIC IMPORTED GLOBAL)
# set (IG_LIBRARIES /work/fhq/Paddle/third_party/ig/libig.so)
# set (IG_LIBRARIES /work/fhq/Paddle/third_party/ig/libig_static.a)
# set_property(TARGET ig PROPERTY IMPORTED_LOCATION ${IG_LIBRARIES})
# add_dependencies(ig ig_lib)
