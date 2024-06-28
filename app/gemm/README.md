# install openBLAS
github commit: 18063b1

cd openBLAS; mkdir install;
sed -i 's/^# NUM_PARALLEL = 2/NUM_PARALLEL = 2/g' Makefile.rule;
sed -i 's/^NUM_PARALLEL = 1/NUM_PARALLEL = 2/g' Makefile.system;
# Lassen cannot set NO_AFFINITY = 0, otherwise compilation fails
# sed -i 's/^NO_AFFINITY = 1/NO_AFFINITY = 0/g' Makefile.rule;
# sed -i 's/^NO_AFFINITY = 1/NO_AFFINITY = 0/g' Makefile.system;
git diff | egrep "^\+N" | wc -l # should be 5 if resetting NO_AFFINITY; otherwise 2
rm -rf install/*
# NO_AFFINITY=0
USE_OPENMP=1 NUM_PARALLEL=2 make -j;
make -j PREFIX=$(realpath install) install
