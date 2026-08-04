# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# POSIX-portable (`?=` only, no `ifeq`/`.if`): see tests/c/Makefile's header
# for why -- GNU make, BSD make, and mingw32-make share no conditional
# directive syntax. Windows overrides (CC, BIN, RUN, RM_F) come from
# `UniAccurate.nimble`'s `make` invocation as command-line variable
# assignments.
CC     ?= cc
ROOT   := $(CURDIR)/../..
CFLAGS := -I$(ROOT)/include -O2 -Wall -Wextra -std=c11
LIB    := $(ROOT)/libUniAccurate.a
BIN    ?= demo
RUN    ?= ./$(BIN)
RM_F   ?= rm -f

.PHONY: all run clean
all: run
run: $(BIN)
	$(RUN)
$(BIN): demo.c $(ROOT)/include/UniAccurate.h $(LIB)
	$(CC) $(CFLAGS) -o $(BIN) demo.c $(LIB)
# File target, not phony: `nimble cexample` already built the lib, don't redo it.
$(LIB):
	cd $(ROOT) && nimble clibStatic
clean:
	$(RM_F) $(BIN)
