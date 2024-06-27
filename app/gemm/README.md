# install openBLAS
```
cd openBLAS; mkdir install;
sed -i 's/^# NUM_PARALLEL = 2/NUM_PARALLEL = 2/g' Makefile.rule;
sed -i 's/^NUM_PARALLEL = 1/NUM_PARALLEL = 2/g' Makefile.system;
sed -i 's/^NO_AFFINITY = 1/NO_AFFINITY = 0/g' Makefile.rule;
sed -i 's/^NO_AFFINITY = 1/NO_AFFINITY = 0/g' Makefile.system;
git diff | egrep "^\+N" | wc -l # should be 5
rm -rf install/*
USE_OPENMP=1 NUM_PARALLEL=2 NO_AFFINITY=0 make -j;
make -j PREFIX=$(realpath install) install
```