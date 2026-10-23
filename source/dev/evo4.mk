# Modified for JustFit: product identity and integration.
# Build with: make -f Makefile -f dev/evo4.mk evo4
# The normal library retains the upstream eight-row ABI; this companion
# contains only operators whose target-verification width changes.
EVO4_KERNELS := gdn draft attention_qkv paged_attention_store tq4_store tq4_attention tq4_rotated sampling
EVO4_AIRS := $(addprefix $(BUILD)/metal/evo4/,$(addsuffix .air,$(EVO4_KERNELS)))
EVO4_LIBRARY := $(BUILD)/justfit-verify16.metallib
.PHONY: evo4
evo4: $(TARGET) $(EVO4_LIBRARY)

$(BUILD)/metal/evo4/%.air: runtime/metal/kernels/decode/%.metal $(KERNEL_HEADERS)
	@mkdir -p $(@D)
	$(METAL) $(PROD_METALFLAGS) -DJUSTFIT_TARGET_VERIFY_ROWS=16u -c $< -o $@

$(EVO4_LIBRARY): $(EVO4_AIRS)
	$(METALLIB) $^ -o $@
