find_package(Sphinx REQUIRED)

add_custom_target(docs)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# Collect all RST files in the docs directory
file(GLOB_RECURSE SPHINX_RST_FILES
     ${CMAKE_CURRENT_SOURCE_DIR}/*.rst)

# Also collect other documentation source files
file(GLOB_RECURSE SPHINX_OTHER_FILES
     ${CMAKE_CURRENT_SOURCE_DIR}/*.md
     ${CMAKE_CURRENT_SOURCE_DIR}/*.py
     ${CMAKE_CURRENT_SOURCE_DIR}/_static/*)

# Only regenerate Sphinx when:
# - Any RST file has been updated
# - The Sphinx config has been updated
# - Other documentation files have been updated
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND
                     ${SPHINX_EXECUTABLE} -b html
                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                   DEPENDS
                   ${SPHINX_RST_FILES}
                   ${SPHINX_OTHER_FILES}
                   MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                   COMMENT "Generating documentation with Sphinx")

# Nice named target so we can run the job easily
add_custom_target(Sphinx DEPENDS ${SPHINX_INDEX_FILE})
add_dependencies(docs Sphinx)
