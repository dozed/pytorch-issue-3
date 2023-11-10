cmake_minimum_required(VERSION 3.23, FATAL_ERROR)
project(pytorch_issue_3)

# link target against libraries
function (common_target_link_libraries target)
    target_link_libraries(${target} "${TORCH_LIBRARIES}")
endfunction()

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/Code/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pytorch_issue_3 main.cpp)
common_target_link_libraries(pytorch_issue_3)

