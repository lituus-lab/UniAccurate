# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# Shell-free: `pwd` and `rm` are unavailable under mingw32-make on Windows.
ifeq ($(origin CC),default)
  ifeq ($(OS),Windows_NT)
    CC := gcc
  else
    CC := cc
  endif
endif

ROOT   := $(CURDIR)/../..
CFLAGS := -I$(ROOT)/include -O2 -Wall -Wextra -std=c11
LIB    := $(ROOT)/libUniAccurate.a

ifeq ($(OS),Windows_NT)
  BIN  := test_uniaccurate.exe
  RUN  := $(BIN)
  RM_F := del /q
  LIBS :=
else
  BIN  := test_uniaccurate
  RUN  := ./$(BIN)
  RM_F := rm -f
  # The dot family calls libm `fma`; the static lib leaves it unresolved, so the
  # final link supplies it. (macOS/MinGW pull libm implicitly; harmless there.)
  LIBS := -lm
endif

.PHONY: all run clean
all: run
run: $(BIN)
	$(RUN)
$(BIN): test_uniaccurate.c $(ROOT)/include/UniAccurate.h $(LIB)
	$(CC) $(CFLAGS) -o $(BIN) test_uniaccurate.c $(LIB) $(LIBS)
# File target, not phony: `nimble ctest` already built the lib, don't redo it.
$(LIB):
	cd $(ROOT) && nimble clibStatic
clean:
	$(RM_F) $(BIN)
