#include "engine/MemoryPlan.hpp"
#include "TestModel.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

DeviceCapabilities device(uint64_t workingSet = 12 * kGiB) {
  DeviceCapabilities result;
  result.deviceName = "test";
  result.appleGpuFamily = 9;
  result.macosMajor = 26;
  result.macosMinor = 4;
  result.physicalMemoryBytes = 16 * kGiB;
  result.recommendedMaxWorkingSetBytes = workingSet;
  result.maxBufferLengthBytes = 8 * kGiB;
  result.maxThreadgroupMemoryBytes = 32 * 1024;
  result.maxThreadgroupWidth = 1024;
  result.hasUnifiedMemory = true;
  result.supportsPlacementSparse = true;
  return result;
}

ModelMemoryProfile model() {
  return test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB);
}

void testUnifiedElasticBudget() {
  EngineMemoryPlan plan = requireEngineMemoryPlan(device(), model());
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.maximumBatchWidth == 4 &&
              budget.kvSparseMappingBatchPages == 128 &&
              budget.kvExtentPages == 128,
          "execution geometry did not reach memory planning");
  require(budget.fixedRuntimeBytes == model().fixedRuntimeBytes(),
          "active state was incorrectly precharged as fixed memory");
  require(budget.dynamicBudgetBytes ==
              budget.hardBudgetBytes - budget.fixedRuntimeBytes,
          "state and KV do not share one dynamic budget");
  require(budget.minimumDynamicBytes ==
                  budget.activeStateCellBytes + budget.kvExtentBytes &&
              budget.minimumRequiredBytes ==
                  budget.fixedRuntimeBytes + budget.minimumDynamicBytes,
          "minimum B1 plus one physical extent is incorrect");
  require(budget.kvVirtualPages % 128 == 0 && budget.kvVirtualPages >= 128 &&
              budget.kvVirtualBytes ==
                  uint64_t{budget.kvVirtualPages} * budget.kvPageBytes &&
              budget.kvVirtualTokens == uint64_t{budget.kvVirtualPages} * 32,
          "KV virtual address space is inconsistent");
  require(plan.maximumContextTokens() ==
              std::min<uint64_t>(model().maximumContextTokens,
                                 budget.kvVirtualTokens -
                                     model::ExecutionLimits::speculativeScratchTokens),
          "advertised context exceeds elastic KV capacity");
  const std::string json = plan.toStatusJson();
  require(json.find("\"dynamic_budget_bytes\"") != std::string::npos &&
              json.find("\"kv_extent_pages\":128") != std::string::npos,
          "elastic state/KV budget is missing from memory status");
}

void testUserCeilingAndFailure() {
  EngineMemoryPlan automatic = requireEngineMemoryPlan(device(), model());
  const uint64_t ceiling =
      automatic.breakdown().minimumRequiredBytes + 64 * kMiB;
  EngineMemoryPlan limited =
      requireEngineMemoryPlan(device(), model(), ceiling);
  require(limited.breakdown().hardBudgetBytes == ceiling,
          "explicit memory ceiling was ignored");
  require(requireEngineMemoryPlan(device(), model(), 16 * kGiB)
                  .breakdown().hardBudgetBytes ==
              automatic.breakdown().hardBudgetBytes,
          "explicit memory ceiling overrode the safe working set");

  auto failed = evaluateEngineMemoryPlan(
      device(), model(), automatic.breakdown().minimumRequiredBytes - 1);
  require(!failed.plan &&
              failed.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "budget smaller than B1 plus one extent was accepted");
}

void testHardBudgetBoundaries() {
  require(EngineMemoryPolicy::hardBudgetBytes(12 * kGiB) == 11 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 8 * kGiB) ==
                  8 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 16 * kGiB) ==
                  11 * kGiB,
          "preflight ceiling disagrees with automatic or explicit policy");
  require(EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 11 * kGiB + 512 * kMiB) ==
              11 * kGiB + 512 * kMiB,
          "an explicit sub-working-set ceiling could not use automatic margin");
  require(EngineMemoryPolicy::hardBudgetBytes(0) == 0 &&
              EngineMemoryPolicy::hardBudgetBytes(kGiB, 1) == 0,
          "insufficient working set underflowed the preflight ceiling");
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  require(EngineMemoryPolicy::hardBudgetBytes(maximum, maximum) ==
              maximum - EngineMemoryPolicy::workingSetMarginBytes(maximum),
          "maximum working set overflowed the preflight ceiling");
}

void testModelProvidedKvGeometry() {
  ModelMemoryProfile compact = model();
  compact.name = "compact-test-model";
  compact.targetKvLayout = {10, 2, 256};
  EngineMemoryPlan plan = requireEngineMemoryPlan(device(), compact);
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.kvPageBytes == 332'800 &&
              budget.kvSparseMappingBatchPages == 256 &&
              budget.kvExtentPages == 512,
          "memory plan ignored model-provided Q8 geometry");
  require(plan.toStatusJson().find("\"attention_layers\":10") !=
              std::string::npos &&
              plan.toStatusJson().find("\"kv_heads\":2") !=
                  std::string::npos,
          "model-provided Q8 geometry is missing from status");
}

void testTq4FullNativeWindowGeometry() {
  ModelMemoryProfile compact = model();
  compact.name = "tq4-full-window-test";
  compact.tq4Kv = true;
  compact.maximumContextTokens = 256 * 1024;

  constexpr uint32_t virtualPages = 8200;
  const uint64_t exactBudget = compact.fixedRuntimeBytes() +
                               compact.footprint.activeStateCellBytes +
                               uint64_t{virtualPages} * compact.kvPageBytes();
  DeviceCapabilities tq4Device = device(32 * kGiB);
  tq4Device.physicalMemoryBytes = 64 * kGiB;
  EngineMemoryPlan plan =
      requireEngineMemoryPlan(tq4Device, compact, exactBudget);
  const auto &budget = plan.breakdown();
  require(budget.kvPageBytes == 532'480 &&
              budget.kvSparseMappingBatchPages == 8 &&
              budget.kvVirtualPages == virtualPages &&
              budget.kvVirtualBytes ==
                  uint64_t{virtualPages} * compact.kvPageBytes(),
          "TQ4 page-major geometry or sparse rounding is incorrect");
  require(plan.maximumContextTokens() == 256 * 1024,
          "TQ4 plan did not cover the full native window plus scratch");
  const std::string json = plan.toStatusJson();
  require(json.find("\"kv_quantization_bits\":4") != std::string::npos,
          "TQ4 memory plan is missing its public format identity");
}

void testPhaseManagedVisionMayBeNonresident() {
  ModelMemoryProfile resident = model();
  resident.footprint.visionWeightsBytes = 0;
  auto rejected = evaluateEngineMemoryPlan(device(), resident);
  require(!rejected.plan &&
              rejected.status.code == BudgetErrorCode::InvalidModelSpec,
          "missing resident vision weights were accepted");

  ModelMemoryProfile phaseManaged = resident;
  phaseManaged.footprint.phaseManagedVision = true;
  EngineMemoryPlan plan = requireEngineMemoryPlan(device(), phaseManaged);
  require(plan.breakdown().visionWeightsBytes == 0,
          "phase-managed vision was charged as resident memory");
  require(plan.maximumContextTokens() > 0,
          "phase-managed vision left no usable KV capacity");
}

void testPhaseManagedDraftAndPrefillDoNotOverlap() {
  ModelMemoryProfile resident = model();
  const uint64_t residentBytes = resident.fixedRuntimeBytes();
  resident.footprint.phaseManagedDraft = true;
  require(resident.fixedRuntimeBytes() ==
              residentBytes - std::min(resident.footprint.draftWeightsBytes,
                                       resident.footprint.sharedPrefillBytes),
          "phase-managed draft did not remove the impossible prefill overlap");
  EngineMemoryPlan plan = requireEngineMemoryPlan(device(), resident);
  require(plan.toStatusJson().find("\"phase_managed_draft\":true") !=
              std::string::npos,
          "phase-managed draft policy is missing from status");
}

} // namespace

void testDeviceValidationNamesTheMacosFloor() {
  require(!device().validationError(),
          "the reference device reported a validation error");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "macOS 26.3 was not refused with the macOS reason");
  // The operating system explains a missing placement-sparse query, so its
  // reason takes precedence over the feature's own.
  older.supportsPlacementSparse = false;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "an older macOS did not take precedence over the sparse reason");
  require(older.macosVersion() == "26.3.0",
          "the macOS version string is not major.minor.patch");
  DeviceCapabilities unknown = device();
  unknown.macosMajor = 0;
  unknown.macosMinor = 0;
  require(unknown.validationError().value_or("") == "macos_26_4_required",
          "an unknown macOS version was accepted");
  DeviceCapabilities newer = device();
  newer.macosMajor = 27;
  newer.macosMinor = 0;
  require(!newer.validationError(), "a newer macOS major was refused");
}

int main() {
  try {
    testUnifiedElasticBudget();
    testUserCeilingAndFailure();
    testHardBudgetBoundaries();
    testModelProvidedKvGeometry();
    testTq4FullNativeWindowGeometry();
    testPhaseManagedVisionMayBeNonresident();
    testPhaseManagedDraftAndPrefillDoNotOverlap();
    testDeviceValidationNamesTheMacosFloor();
    std::cout << "elastic memory plan tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "elastic memory plan tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
