#
# Add the executable
#

ADD_EXECUTABLE(GenDistProb GenerateDistProb.cpp)
TARGET_LINK_LIBRARIES(GenDistProb zxhImageModelings zxhImages zxhCommon ${NIFTI_NIFTILIB_NAME} )
 
INSTALL_TARGETS(/bin/zxh GenDistProb) 
