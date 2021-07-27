#
# Add the executable
#

ADD_EXECUTABLE(GenMeshWall Generate_MeshWall.cpp STLreader.cpp STLreader.h)
TARGET_LINK_LIBRARIES(GenMeshWall zxhImageModelings zxhImages zxhCommon ${NIFTI_NIFTILIB_NAME} )
 
INSTALL_TARGETS(/bin/zxh GenMeshWall)
