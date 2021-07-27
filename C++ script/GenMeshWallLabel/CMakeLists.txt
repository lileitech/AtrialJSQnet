#
# Add the executable
#

ADD_EXECUTABLE(GenMeshWallLabel Generate_MeshWallLabel.cpp STLreader.cpp STLreader.h)
TARGET_LINK_LIBRARIES(GenMeshWallLabel zxhImageModelings zxhImages zxhCommon ${NIFTI_NIFTILIB_NAME} )
 
INSTALL_TARGETS(/bin/zxh GenMeshWallLabel)
