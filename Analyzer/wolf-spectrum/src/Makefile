#!/usr/bin/make -f
# Makefile for DISTRHO Plugins #
# ---------------------------- #
# Created by falkTX, Christopher Arndt, and Patrick Desaulniers
#

# --------------------------------------------------------------
# Installation directories

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
LIBDIR ?= $(PREFIX)/lib
DSSI_DIR ?= $(LIBDIR)/dssi
LADSPA_DIR ?= $(LIBDIR)/ladspa
ifneq ($(MACOS_OR_WINDOWS),true)
LV2_DIR ?= $(LIBDIR)/lv2
VST_DIR ?= $(LIBDIR)/vst
endif
ifeq ($(MACOS),true)
LV2_DIR ?= /Library/Audio/Plug-Ins/LV2
VST_DIR ?= /Library/Audio/Plug-Ins/VST
endif
ifeq ($(WINDOWS),true)
LV2_DIR ?= $(COMMONPROGRAMFILES)/LV2
VST_DIR ?= $(COMMONPROGRAMFILES)/VST2
endif

USER_DSSI_DIR ?= $(HOME)/.dssi
USER_LADSPA_DIR ?= $(HOME)/.ladspa
ifneq ($(MACOS_OR_WINDOWS),true)
USER_LV2_DIR ?= $(HOME)/.lv2
USER_VST_DIR ?= $(HOME)/.vst
endif
ifeq ($(MACOS),true)
USER_LV2_DIR ?= $(HOME)/Library/Audio/Plug-Ins/LV2
USER_VST_DIR ?= $(HOME)/Library/Audio/Plug-Ins/VST
endif
ifeq ($(WINDOWS),true)
USER_LV2_DIR ?= $(APPDATA)/LV2
USER_VST_DIR ?= $(APPDATA)/VST
endif

# --------------------------------------------------------------
# Project name, used for binaries

NAME = wolf-spectrum

# --------------------------------------------------------------
# Plugin types to build

BUILD_LV2 ?= true
BUILD_VST2 ?= true
BUILD_JACK ?= true
BUILD_DSSI ?= false
BUILD_LADSPA ?= false

# --------------------------------------------------------------
# Files to build

FILES_DSP = \
	Utils/src/Mathf.cpp \
	WolfSpectrumPlugin.cpp

FILES_UI = \
        ../libs/kissfft/kiss_fft.c \
	Utils/src/Mathf.cpp \
        Structures/src/PeakFallSmooth.cpp \
	Structures/src/Margin.cpp \
	Widgets/src/ResizeHandle.cpp \
        Widgets/src/ScrollingTexture.cpp \
	Widgets/src/StatusBar.cpp \
	Widgets/src/MenuWidget.cpp \
        Widgets/src/Spectrogram.cpp \
	WolfSpectrumUI.cpp

# --------------------------------------------------------------
# Do some magic

UI_TYPE = opengl

DPF_PATH = ../dpf
DPF_TARGET_DIR = ../bin
DPF_BUILD_DIR = ../build/$(NAME)

include ../dpf/Makefile.plugins.mk

# --------------------------------------------------------------
# Additional flags

BASE_FLAGS += -I./Resources -I./Structures -I./Widgets -I./Utils -I../libs/kissfft -I../libs/readerwriterqueue

BASE_FLAGS += $(CUSTOM_DEFINES)

# Fix missing GL/gl.h include on FreeBSD
BASE_FLAGS += $(shell $(PKG_CONFIG) --cflags-only-I gl)

LINK_FLAGS += -lpthread

ifeq ($(PARANOID_MODE),true)
BASE_FLAGS += -Wall -Wextra -Werror
endif

# --------------------------------------------------------------
# Enable all selected plugin types

ifeq ($(BUILD_LV2),true)
TARGETS += lv2
endif

ifeq ($(BUILD_VST2),true)
TARGETS += vst
endif

ifeq ($(BUILD_JACK),true)
ifeq ($(HAVE_JACK),true)
TARGETS += jack
endif
endif

ifeq ($(BUILD_DSSI),true)
ifneq ($(MACOS_OR_WINDOWS),true)
ifeq ($(HAVE_DGL),true)
ifeq ($(HAVE_LIBLO),true)
TARGETS += dssi
endif
endif
endif
endif

ifeq ($(BUILD_LADSPA),true)
TARGETS += ladspa
endif

all: $(TARGETS)

install: all
ifeq ($(BUILD_DSSI),true)
ifneq ($(MACOS_OR_WINDOWS),true)
ifeq ($(HAVE_DGL),true)
ifeq ($(HAVE_LIBLO),true)
	@mkdir -p -m755 $(DESTDIR)$(DSSI_DIR) && \
	  install -m755 $(TARGET_DIR)/$(NAME)-dssi$(LIB_EXT) $(DESTDIR)$(DSSI_DIR)
endif
endif
endif
endif
ifeq ($(BUILD_LADSPA),true)
	@mkdir -p -m755 $(DESTDIR)$(LADSPA_DIR) && \
	  install -m755 $(TARGET_DIR)/$(NAME)-ladspa$(LIB_EXT) $(DESTDIR)$(LADSPA_DIR)
endif
ifeq ($(BUILD_VST2),true)
	@mkdir -p -m755 $(DESTDIR)$(VST_DIR) && \
	  install -m755 $(TARGET_DIR)/$(NAME)-vst$(LIB_EXT) $(DESTDIR)$(VST_DIR)
endif
ifeq ($(BUILD_LV2),true)
	@mkdir -p -m755 $(DESTDIR)$(LV2_DIR)/$(NAME).lv2 && \
	install -m755 $(TARGET_DIR)/$(NAME).lv2/*$(LIB_EXT) $(DESTDIR)$(LV2_DIR)/$(NAME).lv2 && \
	  install -m644 $(TARGET_DIR)/$(NAME).lv2/*.ttl $(DESTDIR)$(LV2_DIR)/$(NAME).lv2
endif
ifeq ($(BUILD_JACK),true)
ifeq ($(HAVE_JACK),true)
	@mkdir -p -m755 $(DESTDIR)$(BINDIR) && \
          install -m755 $(TARGET_DIR)/$(NAME)$(APP_EXT) $(DESTDIR)$(BINDIR)
endif
endif

install-user: all
ifeq ($(BUILD_DSSI),true)
ifneq ($(MACOS_OR_WINDOWS),true)
ifeq ($(HAVE_DGL),true)
ifeq ($(HAVE_LIBLO),true)
	@mkdir -p -m755 $(USER_DSSI_DIR) && \
          install -m755 $(TARGET_DIR)/$(NAME)-dssi$(LIB_EXT) $(USER_DSSI_DIR)
endif
endif
endif
endif
ifeq ($(BUILD_LADSPA),true)
	@mkdir -p -m755 $(USER_LADSPA_DIR) && \
          install -m755 $(TARGET_DIR)/$(NAME)-ladspa$(LIB_EXT) $(USER_LADSPA_DIR)
endif
ifeq ($(BUILD_VST2),true)
	@mkdir -p -m755 $(USER_VST_DIR) && \
          install -m755 $(TARGET_DIR)/$(NAME)-vst$(LIB_EXT) $(USER_VST_DIR)
endif
ifeq ($(BUILD_LV2),true)
	@mkdir -p -m755 $(USER_LV2_DIR)/$(NAME).lv2 && \
          install -m755 $(TARGET_DIR)/$(NAME).lv2/*$(LIB_EXT) $(USER_LV2_DIR)/$(NAME).lv2 && \
          install -m644 $(TARGET_DIR)/$(NAME).lv2/*.ttl $(USER_LV2_DIR)/$(NAME).lv2
endif
ifeq ($(BUILD_JACK),true)
ifeq ($(HAVE_JACK),true)
	@mkdir -p -m755 $(HOME)/bin && \
          install -m755 $(TARGET_DIR)/$(NAME)$(APP_EXT) $(HOME)/bin
endif
endif

# --------------------------------------------------------------

.PHONY: all install install-user
