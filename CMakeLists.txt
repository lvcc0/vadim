cmake_minimum_required(VERSION 3.16.0)

project(vadim)

set(CMAKE_CXX_STANDARD 20)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)

find_program(SLANGC_EXECUTABLE NAMES slangc REQUIRED)

function (add_slang_shader_target TARGET)
    cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})
    
    set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/res/shaders)
    set(ENTRY_POINTS -entry vertMain -entry fragMain)

    add_custom_command(
        OUTPUT ${SHADERS_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )

    add_custom_command(
        OUTPUT ${SHADERS_DIR}/slang.spv
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
        WORKING_DIRECTORY ${SHADERS_DIR}
        DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
        COMMENT "compiling slang shaders"
        VERBATIM
    )

    add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

add_library(VulkanCppModule INTERFACE)
add_library(Vulkan::cppm ALIAS VulkanCppModule)
target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
target_compile_definitions(VulkanCppModule INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)

add_slang_shader_target(slang_compiled_shaders SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/base.slang)
add_executable("${CMAKE_PROJECT_NAME}" src/main.cpp)
add_dependencies("${CMAKE_PROJECT_NAME}" slang_compiled_shaders)

set_target_properties("${CMAKE_PROJECT_NAME}" PROPERTIES CXX_STANDARD 20)
target_link_libraries("${CMAKE_PROJECT_NAME}" Vulkan::cppm glm glfw)

