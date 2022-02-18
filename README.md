# Bidomain_Compiled
Bidomain Compiled Code

To compile and use it you need an installation of libMesh.

Clone the repository in a new folder.

Denote with LIBMESH_PATH the path to the libMesh installation folder.
Inside LIBMESH_PATH you should have a bin folder with the executable 'libmesh-config'

Once you have found what LIBMESH_PATH is, copy the file build.sh 
$ cp build.sh build_josue.sh
and edit the build_josue.sh file to fill the LIBMESH_PATH variable.

Compile calling
$ sh build.sh

Run calling
$ bidomain -i input_file_name



