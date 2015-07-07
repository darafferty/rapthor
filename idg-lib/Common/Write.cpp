#include "Write.h"

void writeSubgrid(void *_subgrid, const char *name) {
    SubGridType *subgrid = (SubGridType *) _subgrid;
    
    for (int bl = 0; bl < 5000/*NR_BASELINES*/; bl += 1000) {
        char filename[strlen(name) + 1];
        sprintf(filename, "%s_%d", name, bl);

        FILE *file = fopen(filename, "w");

        for (int y = 0; y < SUBGRIDSIZE; y++) {
            for (int x = 0; x < SUBGRIDSIZE; x++) {
                #if ORDER == ORDER_BL_V_U_P
                FLOAT_COMPLEX value = (*subgrid)[0][bl][y][x][0];
                #elif ORDER == ORDER_BL_P_V_U
                FLOAT_COMPLEX value = (*subgrid)[0][bl][0][y][x];
                #endif
                if (x > 0) {
                    fprintf(file, "\t");
                }
                fprintf(file, "%f", value.real());
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }
}

void writeGrid(void *_grid, const char *name) {
    GridType *grid = (GridType *) _grid;
    FILE *file = fopen(name, "w");

    for (int y = 0; y < GRIDSIZE; y++) {
        for (int x = 0; x < GRIDSIZE; x++) {
            FLOAT_COMPLEX value = (*grid)[0][y][x];
            if (x > 0) {
            fprintf(file, "\t");
            }
            fprintf(file, "%.2f", round(value.real()));
        }   
        fprintf(file, "\n");
    }   
    
    fprintf(file, "\n");
    fclose(file);
}

void writeVisibilities(void *_visibilities, const char *name) {
    VisibilitiesType *visibilities = (VisibilitiesType *) _visibilities;
    FILE *file = fopen(name, "w");
    
    //for (int bl = 0; bl < NR_BASELINES; bl++) {
    int bl = NR_BASELINES / 4;
        //for (int time = 0; time < NR_TIME; time++) {
         int time = NR_TIME / 2;
            for (int chan = 0; chan < NR_CHANNELS; chan++) {
                FLOAT_COMPLEX value =(*visibilities)[bl][time][chan][0];
                fprintf(file, "%f + %f I\n", value.real(), value.imag());
            }
        //}
    //}

    fclose(file);
}
