/*
    Read the RAPL values from the sysfs powercap interface

    Original code by Vince Weaver (vincent.weaver@maine.edu)
    http://web.eece.maine.edu/~vweaver/projects/rapl/rapl-read.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "rapl-read.h"

void init_rapl() {
    // Detect cpu and packages
	detect_cpu();
	detect_packages();

    // Check whether rapl works
    #if defined(DEBUG)
    rapl_sysfs();
    if (!rapl_supported) {
        fprintf(stderr, "RAPL not supported, power measurement will be unavailable.\n");
    }
    #endif
}

void print_cpu_model(int model) {
	printf("CPU model: ");

	switch(model) {
		case CPU_SANDYBRIDGE:
			printf("Sandybridge");
			break;
		case CPU_SANDYBRIDGE_EP:
			printf("Sandybridge-EP");
			break;
		case CPU_IVYBRIDGE:
			printf("Ivybridge");
			break;
		case CPU_IVYBRIDGE_EP:
			printf("Ivybridge-EP");
			break;
		case CPU_HASWELL:
			printf("Haswell");
			break;
		case CPU_HASWELL_EP:
			printf("Haswell-EP");
			break;
		case CPU_BROADWELL:
			printf("Broadwell");
			break;
        case CPU_KNIGHTSLANDING:
            printf("Knights Landing");
            break;
		default:
			printf("Unsupported model %d\n",model);
			model=-1;
			break;
	}

	printf("\n");
}

int detect_cpu(void) {
	FILE *fff;
	int family,model=-1;
	char buffer[BUFSIZ],*result;
	char vendor[BUFSIZ];

	fff=fopen("/proc/cpuinfo","r");
	if (fff==NULL) return -1;

	while(1) {
		result=fgets(buffer,BUFSIZ,fff);
		if (result==NULL) break;

		if (!strncmp(result,"vendor_id",8)) {
			sscanf(result,"%*s%*s%s",vendor);

			if (strncmp(vendor,"GenuineIntel",12)) {
				printf("%s not an Intel chip\n",vendor);
				return -1;
			}
		}

		if (!strncmp(result,"cpu family",10)) {
			sscanf(result,"%*s%*s%*s%d",&family);
			if (family!=6) {
				printf("Wrong CPU family %d\n",family);
				return -1;
			}
		}

		if (!strncmp(result,"model",5)) {
			sscanf(result,"%*s%*s%d",&model);
		}
	}

	fclose(fff);

    #if defined(DEBUG)
    print_cpu_model(model);
    #endif

	cpu_model = model;
}

void print_packages() {
	printf("\tDetected %d cores in %d packages\n\n",
		total_cores,total_packages);
}

void detect_packages() {
	char filename[BUFSIZ];
	FILE *fff;
	int package;
	int i;

	for(i=0;i<MAX_PACKAGES;i++) package_map[i]=-1;

	for(i=0;i<MAX_CPUS;i++) {
		sprintf(filename,"/sys/devices/system/cpu/cpu%d/topology/physical_package_id",i);
		fff=fopen(filename,"r");
		if (fff==NULL) break;
		fscanf(fff,"%d",&package);
		fclose(fff);

		if (package_map[package]==-1) {
			total_packages++;
			package_map[package]=i;
		}

	}

	total_cores=i;

    #if defined(DEBUG)
    print_packages();
    #endif
}

PowerSensor::State rapl_sysfs() {
	char event_names[MAX_PACKAGES][NUM_RAPL_DOMAINS][256];
	char filenames[MAX_PACKAGES][NUM_RAPL_DOMAINS][256];
	char basename[MAX_PACKAGES][256];
	char tempfile[256];
	long long measurements[MAX_PACKAGES][NUM_RAPL_DOMAINS];
	int valid[MAX_PACKAGES][NUM_RAPL_DOMAINS];
	int i,j;
	FILE *fff;

    PowerSensor::State state;
    state.timeAtRead = omp_get_wtime();
    state.consumedEnergyPKG  = 0;
    state.consumedEnergyDRAM = 0;

    if (!rapl_supported) {
        return state;
    }

	/* /sys/class/powercap/intel-rapl/intel-rapl:0/ */
	/* name has name */
	/* energy_uj has energy */
	/* subdirectories intel-rapl:0:0 intel-rapl:0:1 intel-rapl:0:2 */
	for(j=0;j<total_packages;j++) {
		i=0;
		sprintf(basename[j],"/sys/class/powercap/intel-rapl/intel-rapl:%d",
			j);
		sprintf(tempfile,"%s/name",basename[j]);
		fff=fopen(tempfile,"r");
		if (fff==NULL) {
			//fprintf(stderr,"\tCould not open %s\n",tempfile);
            rapl_supported = false;
            return state;
		}
		fscanf(fff,"%s",event_names[j][i]);
		valid[j][i]=1;
		fclose(fff);
		sprintf(filenames[j][i],"%s/energy_uj",basename[j]);

		/* Handle subdomains */
		for(i=1;i<NUM_RAPL_DOMAINS;i++) {
			sprintf(tempfile,"%s/intel-rapl:%d:%d/name",
				basename[j],j,i-1);
			fff=fopen(tempfile,"r");
			if (fff==NULL) {
				//fprintf(stderr,"\tCould not open %s\n",tempfile);
				valid[j][i]=0;
				continue;
			}
			valid[j][i]=1;
			fscanf(fff,"%s",event_names[j][i]);
			fclose(fff);
			sprintf(filenames[j][i],"%s/intel-rapl:%d:%d/energy_uj",
				basename[j],j,i-1);

		}
	}

	/* Gather measurements */
	for(j=0;j<total_packages;j++) {
		for(i=0;i<NUM_RAPL_DOMAINS;i++) {
			if (valid[j][i]) {
				fff=fopen(filenames[j][i],"r");
				if (fff==NULL) {
					fprintf(stderr,"\tError opening %s!\n",filenames[j][i]);
				}
				else {
					fscanf(fff,"%lld",&measurements[j][i]);
					fclose(fff);
				}
			}
		}
	}

    // Compute package and dram power consumption from individual measurements
    for(j=0;j<total_packages;j++) {
		for(i=0;i<NUM_RAPL_DOMAINS;i++) {
			if (valid[j][i]) {
                char *event_name = event_names[j][i];
                long long value = measurements[j][i] * 1e-6;

                // RAPL_DOMAIN 0: package
                if (i == 0) {
                    state.consumedEnergyPKG += value;

                // RAPL_DOMAIN 1: dram
                } else if(i == 1) {
                    // Correct for energy unit
                    if (cpu_model == CPU_HASWELL_EP ||
                        cpu_model == CPU_KNIGHTSLANDING) {
                        value *= 0.25;
                    }
                    state.consumedEnergyDRAM += value;
                }
            }
        }
    }

	return state;
}
