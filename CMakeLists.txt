cmake_minimum_required(VERSION 2.8.4)
#to have different projects, put them here. Name your project
project(sandbox)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Then put a makefile in the directory listed after -f,
# and the directory after -C is where your file is
add_custom_target(sandbox COMMAND make -f ${sandbox_SOURCE_DIR}/hardware/KeystoneCode/codeMake
        -C ${sandbox_SOURCE_DIR}/hardware/KeystoneCode CLION_EXE_DIR=${PROJECT_BINARY_DIR}/sandbox)

#Feel free to append this however you like.

project(sandboxmake2)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Then put a makefile in the directory listed after -f,
# and the directory after -C is where your file is
add_custom_target(sandboxmake2 COMMAND make -f ${sandboxmake2_SOURCE_DIR}/hardware/KeystoneCode/DataHandling/sandboxMake
        -C ${sandboxmake2_SOURCE_DIR}/hardware/KeystoneCode/DataHandling CLION_EXE_DIR=${PROJECT_BINARY_DIR}/sandboxmake2)


project(makeall)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_custom_target(makeall COMMAND make -f ${makeall_SOURCE_DIR}/hardware/KeystoneCode/codeMake2
        -C ${makeall_SOURCE_DIR}/hardware/KeystoneCode CLION_EXE_DIR=${PROJECT_BINARY_DIR}/makeall)
