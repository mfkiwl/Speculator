# copy files to os x installer folder tree "/package/resources"
CONFIGURE_FILE(conclusion.html.in ${CMAKE_BINARY_DIR}/package/resources/conclusion.html @ONLY)
CONFIGURE_FILE(welcome.html.in ${CMAKE_BINARY_DIR}/package/resources/welcome.html)
FILE(COPY license.html DESTINATION ${CMAKE_BINARY_DIR}/package/resources/)
CONFIGURE_FILE(distribution.xml.in ${CMAKE_BINARY_DIR}/package/distribution.xml @ONLY)
CONFIGURE_FILE(postinstall.in ${CMAKE_BINARY_DIR}/package/scripts/postinstall @ONLY)

# copy installer generation script
CONFIGURE_FILE(generate_osx_installer.sh.in ${CMAKE_BINARY_DIR}/generate_osx_installer.sh @ONLY)

# copy libsndfile precompiled shared lib 1.0.28
INSTALL(FILES  libsndfile_1.0.28.dylib DESTINATION bin)

# copy gcc related shared libraries
INSTALL(FILES  /usr/local/opt/gcc/lib/gcc/7/libstdc++.6.dylib DESTINATION bin)
INSTALL(FILES  /usr/local/opt/gcc/lib/gcc/7/libgomp.1.dylib DESTINATION bin)
INSTALL(FILES  /usr/local/lib/gcc/7/libgcc_s.1.dylib DESTINATION bin)
