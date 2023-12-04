
# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} XFT_SRCS)


set(XFT_SRCS layers/matmul_helper.h xft.cc layers/rope_2d.cpp layers/rotary_embedding.cpp)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fopenmp -mavx512f -mavx512bw -mavx512vl -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-sign-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-free-nonheap-object -Wno-nonnull")

cc_library(
  xft
  SRCS ${XFT_SRCS}
  )

add_subdirectory(util)
add_subdirectory(kernels)

add_executable(demo test_kernels.cc)

target_link_libraries(demo PUBLIC xdnn xft_utils xft_kernels gtest)


