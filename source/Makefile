# Native SDK: production compiler recipes match the qualified development build.
XCRUN := xcrun
CXX := $(XCRUN) -sdk macosx clang++
METAL := $(XCRUN) -sdk macosx metal
METALLIB := $(XCRUN) -sdk macosx metallib
AR := $(XCRUN) -sdk macosx ar
JUSTFIT_BUILD_HELPER ?= ../rust/frontend/target/release/justfit-rs build-helper
BUILD := build
TARGET := $(BUILD)/justfit-engine
METAL_BUILD := $(BUILD)/metal
# Production kernels are grouped by execution phase under
# runtime/metal/kernels/{prefill,decode,shared}; every .metal file there is
# compiled into the metallib. Templates both phases instantiate live in
# kernels/common/, the host/shader contract in runtime/metal/abi/.
PRODUCTION_KERNEL_SOURCES := $(sort $(wildcard \
	runtime/metal/kernels/prefill/*.metal \
	runtime/metal/kernels/decode/*.metal \
	runtime/metal/kernels/shared/*.metal))
PRODUCTION_KERNEL_NAMES := \
	$(patsubst runtime/metal/kernels/%.metal,%,$(PRODUCTION_KERNEL_SOURCES))
PRODUCTION_AIRS := $(addprefix $(METAL_BUILD)/, \
	$(addsuffix .air,$(PRODUCTION_KERNEL_NAMES)))
KERNEL_HEADERS := $(sort $(wildcard runtime/metal/abi/*.h \
	runtime/metal/kernels/common/*.h))
# macOS 26.4 is the tested floor: the engine refuses older systems at
# startup, and every binary and metallib records it. The MPP kernels need
# macOS 26.2 or newer, and MPP chooses its code path by this target.
MACOS_MIN_VERSION := 26.4
MACOS_TARGET_FLAG := -mmacosx-version-min=$(MACOS_MIN_VERSION)
PROD_METALFLAGS := -std=metal4.0 -O3 -Wall -Wextra -Werror -Iruntime \
	$(MACOS_TARGET_FLAG)
ENGINE_CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime \
	$(MACOS_TARGET_FLAG)
ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
LIB := $(BUILD)/justfit.metallib
.PHONY: all force-build-identity
all: $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(METAL_BUILD): | $(BUILD)
	mkdir -p $@

$(METAL_BUILD)/%.air: runtime/metal/kernels/%.metal $(KERNEL_HEADERS) \
		| $(METAL_BUILD)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(METAL) $(PROD_METALFLAGS) -c $< -o $@

$(LIB): $(PRODUCTION_AIRS)
	$(RUN_CONFIGURED) $(METALLIB) $(BUILD_INPUTS) -o $@

ENGINE_BUILD := $(BUILD)/engine
ENGINE_LIBRARY := $(ENGINE_BUILD)/libjustfit.a
ENGINE_LINKFLAGS := -framework Foundation -framework Metal -framework IOKit \
	-framework IOSurface
ENGINE_DEPFLAGS := -MMD -MP
# Configuration belongs to each successful output, not to a shared timestamp:
# macOS make can treat a new stamp and an old binary in the same second as equal.
BUILD_CONFIG_TOOL := build-config
BUILD_RULES_DIGEST := $(shell shasum -a 256 Makefile $(wildcard dev/native.mk) \
	../rust/frontend/src/build_helpers.rs | shasum -a 256 | cut -c1-16)
shell-quote = '$(subst ','"'"',$(1))'
CONFIG_DIGEST := $(shell printf '%s\0' $(BUILD_RULES_DIGEST) \
	$(call shell-quote,$(CXX)) $(call shell-quote,$(ENGINE_CXXFLAGS)) \
	$(call shell-quote,$(ENGINE_OBJCXXFLAGS)) $(call shell-quote,$(ENGINE_DEPFLAGS)) \
	$(call shell-quote,$(ENGINE_LINKFLAGS)) $(call shell-quote,$(AR)) \
	$(call shell-quote,$(METAL)) $(call shell-quote,$(PROD_METALFLAGS)) \
	$(call shell-quote,$(METALLIB)) | shasum -a 256 | cut -c1-16)
RUN_CONFIGURED = $(JUSTFIT_BUILD_HELPER) $(BUILD_CONFIG_TOOL) record \
	--output $(call shell-quote,$@) --config $(BUILD_CONFIG) --
BUILD_INPUTS = $(filter-out force-build-config,$^)
BUILD_ID_SCRIPT := build-identity
BUILD_ID_HEADER := $(ENGINE_BUILD)/BuildIdentity.hpp
BUILD_ID_STAMP := $(ENGINE_BUILD)/build-identity.json
PRODUCTION_ENGINE_INPUTS := $(sort $(shell find runtime -type f \
	\( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' \
		-o -name '*.hpp' -o -name '*.m' -o -name '*.mm' \) \
	-print))
BUILD_ID_CONSTANT_ARGS = \
	--constant 'runtime=qwen-hybrid-dflash8' \
	--constant 'engine_cxxflags=$(ENGINE_CXXFLAGS)' \
	--constant 'engine_objcxxflags=$(ENGINE_OBJCXXFLAGS)' \
	--constant 'engine_linkflags=$(ENGINE_LINKFLAGS)' \
	--constant 'production_metalflags=$(PROD_METALFLAGS)'
ENGINE_MAIN_OBJECT := $(ENGINE_BUILD)/main.o
ENGINE_METAL_RUNTIME_OBJECT := $(ENGINE_BUILD)/metal/MetalBackend.o
# MetalBackend with the test seam metal/BackendInstrumentation.hpp declares.
# Only tests and benchmarks link it, ahead of the engine library, whose own
# MetalBackend object the linker then never pulls.
ENGINE_INSTRUMENTED_METAL_OBJECT := $(ENGINE_BUILD)/metal/MetalBackendInstrumented.o
# Program with the fault seam ane/ProgramInstrumentation.hpp declares, linked
# the same way.
ENGINE_INSTRUMENTED_ANE_OBJECT := $(ENGINE_BUILD)/ane/ProgramInstrumented.o
ENGINE_CPP_SOURCES := \
	runtime/ops/AneFfn.cpp \
	runtime/ops/AneFfnCalibration.cpp \
	runtime/ops/AneFfnMeasurement.cpp \
	runtime/ops/DraftAttention.cpp \
	runtime/ops/DraftSelector.cpp \
	runtime/ops/Embedding.cpp \
	runtime/ops/ExecutionPlans.cpp \
	runtime/ops/GDN.cpp \
	runtime/ops/Linear.cpp \
	runtime/ops/HostLinear.cpp \
	runtime/ops/LinearGguf.cpp \
	runtime/ops/MoE.cpp \
	runtime/ops/Normalization.cpp \
	runtime/ops/PagedAttention.cpp \
	runtime/ops/PageStorage.cpp \
	runtime/ops/TQ4ExtentStorage.cpp \
	runtime/ops/RoPE.cpp \
	runtime/ops/RowCopy.cpp \
	runtime/ops/Sampling.cpp \
	runtime/ops/Vision.cpp \
	runtime/metal/DeviceCapabilities.cpp \
	runtime/engine/MemoryPlan.cpp \
	runtime/engine/AneFfnStartup.cpp \
	runtime/engine/Scheduler.cpp \
	runtime/engine/Cache.cpp \
	runtime/engine/WriteBehind.cpp \
	runtime/engine/Engine.cpp \
	runtime/engine/MemoryGovernor.cpp \
	runtime/engine/MemoryControl.cpp \
	runtime/engine/KvPool.cpp \
	runtime/engine/KvCache.cpp \
	runtime/engine/KvPageTier.cpp \
	runtime/engine/CacheDirectory.cpp \
	runtime/engine/StateCache.cpp \
	runtime/model/DraftContextPlan.cpp \
	runtime/engine/Protocol.cpp \
	runtime/engine/NativeRuntime.cpp \
	runtime/engine/NativeArguments.cpp \
	runtime/engine/FdTransport.cpp \
	runtime/engine/MemoryAudit.cpp \
	runtime/engine/Status.cpp \
	runtime/model/WeightStore.cpp \
	runtime/model/GgufFile.cpp \
	runtime/model/GgufImage.cpp \
	runtime/model/GgufTarget.cpp \
	runtime/model/AffineTarget.cpp \
	runtime/model/AffinePreparation.cpp \
	runtime/model/DraftCheckpoint.cpp \
	runtime/model/WeightSource.cpp \
	runtime/model/WeightImages.cpp \
	runtime/model/GgufPreparation.cpp \
	runtime/model/Qwen3_6Moe.cpp \
	runtime/model/Qwen3_8.cpp \
	runtime/model/QwenVision.cpp \
	runtime/model/VisionPreparation.cpp \
	runtime/model/VisionLoader.cpp \
	runtime/model/QwenTarget.cpp \
	runtime/model/QwenTargetLoader.cpp \
	runtime/model/DFlashDraft.cpp \
	runtime/model/ModelFactory.cpp \
	runtime/model/SlotFile.cpp \
	runtime/model/QwenState.cpp \
	runtime/model/Runtime.cpp \
	runtime/model/RuntimeArenas.cpp
ENGINE_MM_SOURCES := \
	runtime/ane/Handoff.mm \
	runtime/ane/Program.mm \
	runtime/model/SafetensorsCheckpoint.mm \
	runtime/model/AffineDecay.mm \
	runtime/model/ModelDescriptor.mm \
	runtime/engine/RuntimeResources.mm \
	runtime/engine/Bootstrap.mm
ENGINE_OBJECTS := \
	$(patsubst runtime/%.cpp,$(ENGINE_BUILD)/%.o,$(ENGINE_CPP_SOURCES)) \
	$(patsubst runtime/%.mm,$(ENGINE_BUILD)/%.o,$(ENGINE_MM_SOURCES)) \
	$(ENGINE_METAL_RUNTIME_OBJECT)
PRODUCTION_CONFIG_TARGETS := $(ENGINE_OBJECTS) $(ENGINE_MAIN_OBJECT) \
	$(ENGINE_INSTRUMENTED_METAL_OBJECT) $(ENGINE_INSTRUMENTED_ANE_OBJECT) \
	$(ENGINE_LIBRARY) $(PRODUCTION_AIRS) $(LIB) $(TARGET)
ENGINE_DEPFILES := $(ENGINE_OBJECTS:.o=.d) $(ENGINE_MAIN_OBJECT:.o=.d) \
	$(ENGINE_INSTRUMENTED_METAL_OBJECT:.o=.d) $(ENGINE_INSTRUMENTED_ANE_OBJECT:.o=.d)

-include $(ENGINE_DEPFILES)

force-build-identity:

$(ENGINE_BUILD): | $(BUILD)
	mkdir -p $@

$(BUILD_ID_STAMP): force-build-identity | $(ENGINE_BUILD)
	@$(JUSTFIT_BUILD_HELPER) $(BUILD_ID_SCRIPT) write --root . \
		--header $(BUILD_ID_HEADER) --stamp $(BUILD_ID_STAMP) \
		$(BUILD_ID_CONSTANT_ARGS) >/dev/null

$(BUILD_ID_HEADER): $(BUILD_ID_STAMP)
	@:

$(ENGINE_BUILD)/%.o: runtime/%.cpp
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) $(ENGINE_DEPFLAGS) -c $< -o $@

$(ENGINE_BUILD)/%.o: runtime/%.mm
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) -c $< -o $@

$(ENGINE_METAL_RUNTIME_OBJECT): runtime/metal/MetalBackend.mm
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) -c $< -o $@

$(ENGINE_INSTRUMENTED_METAL_OBJECT): runtime/metal/MetalBackend.mm
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) \
		-DJUSTFIT_BACKEND_INSTRUMENTATION=1 -c $< -o $@

$(ENGINE_INSTRUMENTED_ANE_OBJECT): runtime/ane/Program.mm
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) \
		-DJUSTFIT_ANE_INSTRUMENTATION=1 -c $< -o $@

$(ENGINE_MAIN_OBJECT): runtime/main.mm $(BUILD_ID_HEADER)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) \
		-include $(BUILD_ID_HEADER) -c $< -o $@

$(ENGINE_LIBRARY): $(ENGINE_OBJECTS)
	@mkdir -p $(dir $@)
	$(RM) $@
	$(RUN_CONFIGURED) $(AR) rcs $@ $(BUILD_INPUTS)

$(TARGET): $(ENGINE_MAIN_OBJECT) $(ENGINE_LIBRARY) $(LIB) \
		| $(BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_MAIN_OBJECT) $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

.PHONY: force-build-config
force-build-config:
KERNEL_HEADER_NAMES_DIGEST := $(shell printf '%s\0' $(sort $(KERNEL_HEADERS)) | shasum -a 256 | cut -c1-16)
KERNEL_SOURCE_NAMES_DIGEST := $(shell printf '%s\0' $(sort $(PRODUCTION_KERNEL_SOURCES)) | shasum -a 256 | cut -c1-16)
PRODUCTION_AIR_CONFIG := $(CONFIG_DIGEST)-$(KERNEL_HEADER_NAMES_DIGEST)
PRODUCTION_LIB_CONFIG := $(PRODUCTION_AIR_CONFIG)-$(KERNEL_SOURCE_NAMES_DIGEST)
PRODUCTION_CONFIG_TARGETS := $(filter-out $(PRODUCTION_AIRS) $(LIB),$(PRODUCTION_CONFIG_TARGETS))
$(PRODUCTION_CONFIG_TARGETS): BUILD_CONFIG := $(CONFIG_DIGEST)
$(PRODUCTION_AIRS): BUILD_CONFIG := $(PRODUCTION_AIR_CONFIG)
$(LIB): BUILD_CONFIG := $(PRODUCTION_LIB_CONFIG)
STALE_CONFIG_TARGETS := $(shell $(JUSTFIT_BUILD_HELPER) $(BUILD_CONFIG_TOOL) stale --config $(CONFIG_DIGEST) $(PRODUCTION_CONFIG_TARGETS)) \
 $(shell $(JUSTFIT_BUILD_HELPER) $(BUILD_CONFIG_TOOL) stale --config $(PRODUCTION_AIR_CONFIG) $(PRODUCTION_AIRS)) \
 $(shell $(JUSTFIT_BUILD_HELPER) $(BUILD_CONFIG_TOOL) stale --config $(PRODUCTION_LIB_CONFIG) $(LIB))
ifneq ($(strip $(STALE_CONFIG_TARGETS)),)
$(STALE_CONFIG_TARGETS): force-build-config
endif
