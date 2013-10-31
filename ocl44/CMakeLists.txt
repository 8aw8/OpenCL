project(ocl44)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

INCLUDE_DIRECTORIES("./" "include/" "src/" "/opt/AMDAPP/include" "/opt/cuda/include" )

link_directories(/opt/AMDAPP/lib/x86_64)

target_link_libraries(ocl44 /opt/AMDAPP/lib/x86_64/libOpenCL.so)



