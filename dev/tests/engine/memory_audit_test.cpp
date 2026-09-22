#include "TestModel.hpp"
#include "engine/MemoryAudit.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

EngineMemoryPlan plan() {
  DeviceCapabilities device;
  device.deviceName = "test";
  device.appleGpuFamily = 9;
  device.macosMajor = 26;
  device.macosMinor = 4;
  device.physicalMemoryBytes = 32 * kGiB;
  device.recommendedMaxWorkingSetBytes = 24 * kGiB;
  device.maxBufferLengthBytes = 16 * kGiB;
  device.maxThreadgroupMemoryBytes = 32 * 1024;
  device.maxThreadgroupWidth = 1024;
  device.hasUnifiedMemory = true;
  device.supportsPlacementSparse = true;
  return requireEngineMemoryPlan(
      device, test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB));
}

ActualMemoryReport report(const EngineMemoryPlan &memoryPlan) {
  const auto &b = memoryPlan.breakdown();
  ActualMemoryReport result;
  result.targetWeightsBytes = b.targetWeightsBytes;
  result.draftWeightsBytes = b.draftWeightsBytes;
  result.visionWeightsBytes = b.visionWeightsBytes;
  result.stateResidentBytes = b.activeStateCellBytes * 3;
  result.sharedPrefillBytes = b.sharedPrefillBytes;
  result.sharedDecodeBytes = b.sharedDecodeBytes;
  result.kvResidentBytes = b.kvExtentBytes;
  result.backendAllocatedBytes =
      result.targetWeightsBytes + result.draftWeightsBytes +
      result.visionWeightsBytes + result.stateResidentBytes +
      result.sharedPrefillBytes + result.sharedDecodeBytes +
      result.kvResidentBytes;
  result.deviceCurrentAllocatedBytes = result.backendAllocatedBytes;
  result.devicePeakAllocatedBytes = result.backendAllocatedBytes + 16 * kMiB;
  result.estimatedWarmupPeakBytes = result.devicePeakAllocatedBytes;
  return result;
}

void testUnifiedDynamicAudit() {
  EngineMemoryPlan memoryPlan = plan();
  ActualMemoryReport actual = report(memoryPlan);
  auto valid = auditActualMemory(memoryPlan, actual);
  require(valid.valid && valid.error == MemoryAuditError::None,
          "valid elastic memory report failed audit");
  require(valid.toStatusJson().find("\"scope\":\"startup_warmup\"") !=
              std::string::npos,
          "memory audit status does not identify its startup scope");

  ActualMemoryReport overflow = actual;
  overflow.backendAllocatedBytes -= overflow.stateResidentBytes;
  overflow.stateResidentBytes = memoryPlan.breakdown().dynamicBudgetBytes;
  overflow.backendAllocatedBytes += overflow.stateResidentBytes;
  overflow.deviceCurrentAllocatedBytes = overflow.backendAllocatedBytes;
  overflow.devicePeakAllocatedBytes = overflow.backendAllocatedBytes;
  overflow.estimatedWarmupPeakBytes = overflow.backendAllocatedBytes;
  auto rejected = auditActualMemory(memoryPlan, overflow);
  require(!rejected.valid &&
              rejected.error == MemoryAuditError::CategoryExceedsPlan,
          "dynamic state/KV budget overflow was accepted");
}

void testFixedCategoryAndPeakFailures() {
  EngineMemoryPlan memoryPlan = plan();
  ActualMemoryReport actual = report(memoryPlan);
  actual.sharedDecodeBytes = memoryPlan.breakdown().sharedDecodeBytes + 1;
  require(auditActualMemory(memoryPlan, actual).error ==
              MemoryAuditError::CategoryExceedsPlan,
          "fixed arena overflow was accepted");

  actual = report(memoryPlan);
  actual.devicePeakAllocatedBytes = memoryPlan.breakdown().hardBudgetBytes + 1;
  actual.estimatedWarmupPeakBytes = actual.devicePeakAllocatedBytes;
  require(auditActualMemory(memoryPlan, actual).error ==
              MemoryAuditError::HardBudgetExceeded,
          "hard budget overflow was accepted");

  actual = report(memoryPlan);
  actual.estimatedWarmupPeakBytes = actual.devicePeakAllocatedBytes / 2;
  require(auditActualMemory(memoryPlan, actual).error ==
              MemoryAuditError::WarmupEstimateDeviation,
          "bad warmup estimate was accepted");
}

void testPhaseManagedVisionMayBeAbsentFromAudit() {
  DeviceCapabilities device;
  device.deviceName = "test";
  device.appleGpuFamily = 9;
  device.macosMajor = 26;
  device.macosMinor = 4;
  device.physicalMemoryBytes = 32 * kGiB;
  device.recommendedMaxWorkingSetBytes = 24 * kGiB;
  device.maxBufferLengthBytes = 16 * kGiB;
  device.maxThreadgroupMemoryBytes = 32 * 1024;
  device.maxThreadgroupWidth = 1024;
  device.hasUnifiedMemory = true;
  device.supportsPlacementSparse = true;

  ModelMemoryProfile profile =
      test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 0);
  profile.footprint.phaseManagedVision = true;
  EngineMemoryPlan memoryPlan = requireEngineMemoryPlan(device, profile);
  ActualMemoryReport actual = report(memoryPlan);
  require(actual.visionWeightsBytes == 0,
          "test fixture unexpectedly retained phase-managed vision");
  auto valid = auditActualMemory(memoryPlan, actual);
  require(valid.valid && valid.error == MemoryAuditError::None,
          "phase-managed nonresident vision failed memory audit");
}

} // namespace

int main() {
  try {
    testUnifiedDynamicAudit();
    testFixedCategoryAndPeakFailures();
    testPhaseManagedVisionMayBeAbsentFromAudit();
    std::cout << "elastic memory audit tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "elastic memory audit tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
