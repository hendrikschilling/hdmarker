# FIXME multiple urls? authentication?
#set(GIT_REPO "git@hci-repo.iwr.uni-heidelberg.de:light-field/fnmatch.git")
set(GIT_REPO "http://hci-repo.iwr.uni-heidelberg.de/light-field/fnmatch.git")

function(vad_system)
  add_library(FNMATCH::FNMATCH INTERFACE IMPORTED GLOBAL)  
  set(fnmatch_FOUND true PARENT_SCOPE)
endfunction()

function(vad_live)
  message("run VAD_LIVE for FNMATCH")
  
  if (WIN32)
  
    git_clone(fnmatch)
    
    # TODO default seems to point to checkout out dir in source?!?
    add_subdirectory("${VAD_EXTERNAL_ROOT}/fnmatch" "${CMAKE_BINARY_DIR}/external/fnmatch")
    
    #add_library(fnmatch INTERFACE IMPORTED)  
    
    set_target_properties(fnmatch PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${VAD_EXTERNAL_ROOT}/fnmatch")
    set_target_properties(fnmatch PROPERTIES INTERFACE_LINK_LIBRARIES fnmatch)
  else()
    message("use system dependency for fnmatch on *NIX!")
  endif()

endfunction()
