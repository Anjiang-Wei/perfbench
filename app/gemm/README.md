# install openBLAS
```
cd openBLAS; mkdir install;
sed -i 's/^# NUM_PARALLEL = 2/NUM_PARALLEL = 2/' Makefile.rule
USE_OPENMP=1 make -j;
make -j PREFIX=$(realpath install) install
```