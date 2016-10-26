#include "PowerSensor.h"

#ifndef RAPL_READ_H
#define RAPL_READ_H

#define MAX_CPUS	        1024
#define MAX_PACKAGES        16
#define NUM_RAPL_DOMAINS	4

/* Intel CPU models */
#define CPU_SANDYBRIDGE		42
#define CPU_SANDYBRIDGE_EP	45
#define CPU_IVYBRIDGE		58
#define CPU_IVYBRIDGE_EP	62
#define CPU_HASWELL		    60	// 69,70 too?
#define CPU_HASWELL_EP		63
#define CPU_BROADWELL		61	// 71 too?
#define CPU_BROADWELL_EP	79
#define CPU_BROADWELL_DE	86
#define CPU_SKYLAKE		    78	// 94 too?
#define CPU_KNIGHTSLANDING  87


static bool rapl_supported = true;
static int cpu_model       = 0;
static int total_cores     = 0;
static int total_packages  = 0;
static int package_map[MAX_PACKAGES];

void init_rapl();
int detect_cpu(void);
void detect_packages();
PowerSensor::State rapl_sysfs();

#endif
