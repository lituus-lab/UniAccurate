# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# POSIX-portable (`?=` only, no `ifeq`/`.if`): GNU make (Linux/macOS),
# mingw32-make (Windows), and BSD make (FreeBSD) all support `?=`, but only
# GNU make understands `ifeq`/`endif` -- BSD make's own conditional syntax
# (`.if`/`.endif`) is not shared with it either, so there is no single
# directive form portable across all three. The Windows-only overrides (CC,
# BIN, RUN, RM_F, LIBS) come from `UniAccurate.nimble`'s `make` invocation as
# command-line variable assignments instead, which take precedence over `?=`
# on every make flavor without needing an in-Makefile OS branch.
CC     ?= cc
ROOT   := $(CURDIR)/../..
CFLAGS := -I$(ROOT)/include -O2 -Wall -Wextra -std=c11
LIB    := $(ROOT)/libUniAccurate.a
BIN    ?= test_uniaccurate
RUN    ?= ./$(BIN)
RM_F   ?= rm -f
# The dot family calls libm `fma`; the static lib leaves it unresolved, so the
# final link supplies it. (macOS/MinGW pull libm implicitly; harmless there.)
LIBS   ?= -lm

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
