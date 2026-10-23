#include <mach/mach.h>
#include <libproc.h>
#include <sys/resource.h>
#include <stdint.h>
int attributed_self(uint64_t *out) {
  task_vm_info_data_t i = {0};
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&i, &n) != KERN_SUCCESS) return 0;
  out[0]=i.phys_footprint;
  out[1]=i.ledger_tag_graphics_footprint;
  out[2]=i.ledger_tag_graphics_nofootprint;
  out[3]=i.ledger_tag_neural_footprint;
  out[4]=i.ledger_tag_neural_nofootprint;
  out[5]=i.ledger_tag_neural_nofootprint_peak;
  out[6]=i.ledger_tag_neural_nofootprint_total;
  return 1;
}
int attributed_pid(int pid, uint64_t *out) {
  struct rusage_info_v6 r = {0};
  if(proc_pid_rusage(pid,RUSAGE_INFO_V6,(rusage_info_t *)&r)) return 0;
  out[0]=r.ri_phys_footprint;
  out[1]=r.ri_neural_footprint;
  out[2]=r.ri_lifetime_max_neural_footprint;
  out[3]=r.ri_lifetime_max_phys_footprint;
  return 1;
}
