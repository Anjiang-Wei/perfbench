# install openBLAS
cd openBLAS; mkdir install;
USE_OPENMP=1 make -j;
make -j PREFIX=$(realpath install) install