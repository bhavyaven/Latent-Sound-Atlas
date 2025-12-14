cmake_minimum_required(VERSION 3.15)
project(LatentSoundAtlas)

if(MSVC AND NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    message(WARNING "Using a non-MSVC generator with Visual Studio may break .sln generation.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Paths for manual libraries
set(GLEW_ROOT "${CMAKE_SOURCE_DIR}/lib/glew")
set(GLFW_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/glfw/include")
set(GLFW_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/glfw/lib-vc2022")


# Link directories
link_directories(
    ${GLEW_ROOT}/lib/Release/x64
    ${GLFW_LIB_DIR}
    ${PROJECT_SOURCE_DIR}/lib/glad/include
    ${PROJECT_SOURCE_DIR}/lib/oscpp/include
    ${PROJECT_SOURCE_DIR}/src
)

# Source files
set(SOURCES
    src/main.cpp
    # src/SoundPoint.cpp
    src/Renderer.cpp
    src/Camera.cpp
    src/Shader.cpp
    src/OSCManager.cpp
    src/DataLoader.cpp
    src/ParticleSystem.cpp
    lib/glad/src/glad.c
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES} "src/camera.h" "src/shader.h" "src/oscmanager.h" "src/renderer.h" "src/particlesystem.h" "src/stb.cpp")

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    src/
    lib/oscpack
    lib/glad/include
    lib/glm
    ${GLEW_ROOT}/include
    ${GLFW_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    opengl32          # Windows OpenGL library
    "${GLFW_LIB_DIR}/glfw3.lib"            
    "${GLEW_ROOT}/lib/Release/x64/glew32.lib"
    ws2_32
   
)

# Copy shaders and data to build directory
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/opengl_data DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})

