#include "Util.h"

#if RW
#include "Types.h"

extern "C" {

void writeSubgrid(SubGridType *subgrid, const char *name) {
    for (int bl = 0; bl < 5000/*NR_BASELINES*/; bl += 1000) {
        char filename[strlen(name) + 1];
        sprintf(filename, "%s_%d", name, bl);

        FILE *file = fopen(filename, "w");

        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                #if ORDER == ORDER_BL_V_U_P
                float complex value = (*subgrid)[0][bl][y][x][0];
                #elif ORDER == ORDER_BL_P_V_U
                float complex value = (*subgrid)[0][bl][0][y][x];
                #endif
                if (x > 0) {
                    fprintf(file, "\t");
                }
                #if 0 
                fprintf(file, "%.2f", round(creal(value)));
                #else
                fprintf(file, "%f", value);
                #endif
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }
}

void writeGrid(GridType *grid, const char *name) {
    FILE *file = fopen(name, "w");

    for (int y = 0; y < GRIDSIZE; y++) {
        for (int x = 0; x < GRIDSIZE; x++) {
            float complex value = (*grid)[0][y][x];
            if (x > 0) {
            fprintf(file, "\t");
            }
            fprintf(file, "%.2f", round(creal(value)));
        }   
        fprintf(file, "\n");
    }   
    
    fprintf(file, "\n");
    fclose(file);
}

void writeVisibilities(VisibilitiesType *visibilities, const char *name) {
    FILE *file = fopen(name, "w");
    
    //for (int bl = 0; bl < NR_BASELINES; bl++) {
    int bl = NR_BASELINES / 4;
        //for (int time = 0; time < NR_TIME; time++) {
         int time = NR_TIME / 2;
            for (int chan = 0; chan < NR_CHANNELS; chan++) {
                float complex value =(*visibilities)[bl][time][chan][0];
                fprintf(file, "%f + %f I\n", creal(value), cimag(value));
            }
        //}
    //}

    fclose(file);
}
}

#else
Util::Util(
    const char *cc, const char *cflags,
    int nr_stations, int nr_baselines, int nr_time, int nr_channels,
    int nr_polarizations, int SUBGRIDSIZE, int gridsize, float imagesize) {
    // Get compile options
	std::string parameters = definitions(
		nr_stations, nr_baselines, nr_time, nr_channels,
		nr_polarizations, SUBGRIDSIZE, gridsize, imagesize);

    // Compile util wrapper
	std::string options_util = parameters + " " +
	                           cflags     + " " +
                               SRC_RW     + " ";
	rw::Source(SRC_UTIL).compile(cc, SO_UTIL, options_util.c_str());
	
	// Load module
	module = new rw::Module(SO_UTIL);
}

void Util::writeSubgrid(void *subgrid, const char *name) {
    ((void (*)(void*,const char*))
    rw::Function(*module, FUNCTION_WRITESUBGRID).get())(
    subgrid, name);
}

void Util::writeGrid(void *grid, const char *name) {
    ((void (*)(void*,const void*))
    rw::Function(*module, FUNCTION_WRITEGRID).get())(
    grid, name);
}

void Util::writeVisibilities(void *visibilities, const char *name) {
    ((void (*)(void*,const char*))
    rw::Function(*module, FUNCTION_WRITEVISIBILITIES).get())(
    visibilities, name);
}
#endif
