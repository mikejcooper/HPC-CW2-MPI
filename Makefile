# Makefile
SHELL:=/bin/bash

EXE=d2q9-bgk
EXE1=collision_mpi
CC=mpicc
CFLAGS= -std=c99 -xHOST -openmp 
LIBS = -lm

MODULE_1 =module load languages/intel-compiler-15
MODULE_2 =module load languages/python-2.7.6

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): 
	$(MODULE_1) && \
	$(CC) $(CFLAGS) $(EXE).c $(EXE1).c $(LIBS) -o $@

check:
	$(MODULE_2) && \
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)


# make && qsub job_submit_d2q9-bgk && make clean && watch qstat -u $USER


