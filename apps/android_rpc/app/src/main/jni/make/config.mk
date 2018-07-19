#-------------------------------------------------------------------------------
#  Template configuration for compiling
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  ./build.sh
#
#-------------------------------------------------------------------------------
APP_ABI = arm64-v8a

APP_PLATFORM = android-17

# whether enable OpenCL during compile
USE_OPENCL = 1

# whether to enable Vulkan during compile
USE_VULKAN = 0

ifeq ($(USE_VULKAN), 1)
  # Statically linking vulkan requires API Level 24 or higher
  APP_PLATFORM = android-24
endif

# the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
ADD_C_INCLUDES = /opt/rocm/opencl/include

# the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
ADD_LDLIBS = ~/build/sdk/platform-tools/mate10pro/libOpenCL.so
