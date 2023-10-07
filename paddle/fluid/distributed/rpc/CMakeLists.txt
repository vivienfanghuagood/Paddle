if(WITH_PYTHON)
  set(PADDLE_RPC_SRCS python_rpc_handler.cc rpc_agent.cc)
  set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-error=unused-value -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
  )
else()
  set(PADDLE_RPC_SRCS rpc_agent.cc)
  set(DISTRIBUTE_COMPILE_FLAGS
    "-DPADDLE_NO_PYTHON -Wno-error=unused-value -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
  )
endif()



if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

if(NOT WITH_PYTHON)
  add_definitions(-DPADDLE_NO_PYTHON)
endif()

if(WITH_PYTHON)
  set_source_files_properties(
    python_rpc_handler.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endif()

set_source_files_properties(rpc_agent.cc PROPERTIES COMPILE_FLAGS
                                                    ${DISTRIBUTE_COMPILE_FLAGS})
if(WITH_PYTHON)
  set(PADDLE_RPC_DEPS
      brpc
      ssl
      crypto
      protobuf
      zlib
      leveldb
      snappy
      gflags
      glog
      pybind)
else()
  set(PADDLE_RPC_DEPS
        brpc
        ssl
        crypto
        protobuf
        zlib
        leveldb
        snappy
        gflags
        glog)
endif()
proto_library(paddle_rpc_proto SRCS rpc.proto)
cc_library(
  paddle_rpc
  SRCS ${PADDLE_RPC_SRCS}
  DEPS ${PADDLE_RPC_DEPS} paddle_rpc_proto)
add_dependencies(paddle_rpc brpc)
