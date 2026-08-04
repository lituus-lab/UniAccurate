# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# POSIX-portable (`?=` only, no `ifeq`/`.if`): see tests/c/Makefile's header
# for why -- GNU make, BSD make, and mingw32-make share no conditional
# directive syntax. Windows overrides (CC, BIN, RUN, RM_F) come from
# `UniAccurate.nimble`'s `make` invocation as command-line variable
# assignments.
# ROOT stays relative: `$(CURDIR)` is a GNU make builtin BSD make leaves empty
# (it spells it `.CURDIR`) -- see tests/c/Makefile.
CC     ?= cc
ROOT   := ../..
CFLAGS := -I$(ROOT)/include -O2 -Wall -Wextra -std=c11
LIB    := $(ROOT)/libUniAccurate.a
BIN    ?= demo
RUN    ?= ./$(BIN)
RM_F   ?= rm -f
# The dot family calls libm `fma`; the static lib leaves it unresolved, so the
# final link supplies it. (macOS/MinGW pull libm implicitly; harmless there.)
LIBS   ?= -lm

.PHONY: all run clean
all: run
run: $(BIN)
	$(RUN)
$(BIN): demo.c $(ROOT)/include/UniAccurate.h $(LIB)
	$(CC) $(CFLAGS) -o $(BIN) demo.c $(LIB) $(LIBS)
# File target, not phony: `nimble cexample` already built the lib, don't redo it.
$(LIB):
	cd $(ROOT) && nimble clibStatic
clean:
	$(RM_F) $(BIN)
