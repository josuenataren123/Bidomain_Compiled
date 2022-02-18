#SETUP YOUR LIBMESH PATH
# example: 
# LIBMESH_PATH=/nas/longleaf/home/srossi/TPL/libmesh/1.5.1/linux-opt-gcc9-mpich
LIBMESH_PATH=

`${LIBMESH_PATH}/bin/libmesh-config --cxx` Bidomain.cpp -o bidomain `${LIBMESH_PATH}/bin/libmesh-config --cppflags --cxxflags --include --libs`

