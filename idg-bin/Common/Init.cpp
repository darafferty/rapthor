#include "Init.h"
#include "Configuration.h"

#if RW
#include "Types.h"

extern "C" {

/*
	Init
*/
void* init_visibilities() {
	VisibilitiesType *visibilities = (VisibilitiesType *) malloc(sizeof(VisibilitiesType));
	
	#if USE_REAL_VISIBILITIES
	// Load real visibilities
	extern float complex realVisibilities[NR_TIME][NR_BASELINES][NR_CHANNELS][NR_POLARIZATIONS];
	
	// Set all visibilities
	for (int bl = 0; bl < NR_BASELINES; bl++) {
		for (int time = 0; time < NR_TIME; time++) {
			for (int chan = 0; chan < NR_CHANNELS; chan++) {
				for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
					(*visibilities)[bl][time][chan][pol] = realVisibilities[time][bl][chan][pol];
				}
			}
		}
	}
	#else
	// Fixed visibility
	float complex visibility = 2 + 1 * I;
	
	// Set all visibilities
	for (int bl = 0; bl < NR_BASELINES; bl++) {
		for (int time = 0; time < NR_TIME; time++) {
			for (int chan = 0; chan < NR_CHANNELS; chan++) {
				for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
					(*visibilities)[bl][time][chan][pol] = visibility;
				}
			}
		}
	}	
	#endif

	return visibilities;
}

void* init_uvw() {
	// Constants for UVW dataset
	const float speed_of_light = 299792458.;
	const int w_planes = 32;
	const int cell_size_u = (1.08*13107.2 / GRIDSIZE);
	const int cell_size_v = (1.08*13107.2 / GRIDSIZE);
	const int cell_size_w = (8192.0 / w_planes);
	
	// Compute scales
	float scale_u = 59908828.7353515625 / (cell_size_u * speed_of_light);
	float scale_v = 59908828.7353515625 / (cell_size_u * speed_of_light);
	float scale_w = 59908828.7353515625 / (cell_size_u * speed_of_light);

	// Load real uvw coordinates
	extern double realUVW[NR_TIME_DATA][NR_BASELINES_DATA][3];
	
	UVW *uvw = (UVW *) malloc(sizeof(UVWType));

    int i = 0;
	for (int bl = 0; bl < NR_BASELINES_DATA; bl++) {
		int mappedBaseline = bl + (int) ((sqrt((double) (bl * 8 + 1) + 1) / 2)); 
		
		// Iterate all timesteps
		for (int time = 0; time < NR_TIME_DATA; time++) {
		    // Load UVW value
			const double *currentUVW = realUVW[time][mappedBaseline];

			// Set uvw values
			uvw[i++] = make_UVW(
				 scale_u * currentUVW[0] + GRIDSIZE / 2.0f - BLOCKSIZE / 2.0f,
				 scale_v * currentUVW[1] + GRIDSIZE / 2.0f - BLOCKSIZE / 2.0f,
				 scale_w * currentUVW[2] + w_planes / 2.0f
			);
			
			// Truncuate
			if (i == NR_BASELINES * NR_TIME) {
    			return (UVWType *) uvw;
			}
		}
	}
	
	return (UVWType *) uvw;
}

void* init_offset() {
	UVWType *uvw = (UVWType *) init_uvw();
	OffsetType *offset = (OffsetType *) malloc(sizeof(OffsetType));
	
	for (int bl = 0; bl < NR_BASELINES; bl++) {
		// Get first and last UVW coordinate
		UVW uvw_first = (*uvw)[bl][0];
		UVW uvw_last  = (*uvw)[bl][NR_TIME-1];

		// Compute average coordinates
		float u_average = ((uvw_first.u + uvw_last.u) / 2);
		float v_average = ((uvw_first.v + uvw_last.v) / 2);
		float w_average = ((uvw_first.w + uvw_last.w) / 2);
	
		// Set offset relative to zenith
		int u = (GRIDSIZE/2) - u_average;
		int v = (GRIDSIZE/2) - v_average;
		int w = (GRIDSIZE/2) - w_average;
		(*offset)[bl] = make_UVW(u, v, w);
	}
	
	free(uvw);
	
	return offset;
}

void* init_wavenumbers() {
	WavenumberType *wavenumbers = (WavenumberType *) malloc(sizeof(WavenumberType));
	
	// Initialize frequencies
	float frequencies[NR_CHANNELS];
	for (int chan = 0; chan < NR_CHANNELS; chan++) {
		frequencies[chan] = 59908828.7353515625 + 12207.03125 * chan;
	}
	
	const float speed_of_light = 299792458.;
	
	// Initialize wavenumbers
	for (int i = 0; i < NR_CHANNELS; i++) {
		(*wavenumbers)[i] =  2 * M_PI * frequencies[i] / speed_of_light;
	}
	
	return wavenumbers;
}

void* init_aterm() {
	ATermType *aterm = (ATermType *) malloc(sizeof(ATermType));
	
	float complex value = 1 + 1 * I;
	
	for (int ant = 0; ant < NR_STATIONS; ant++) {
		for (int y = 0; y < BLOCKSIZE; y++) {
			for (int x = 0; x < BLOCKSIZE; x++) {
				for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
					(*aterm)[ant][pol][y][x] = value;
				}
			}
		}
	}
	
	return aterm;
}

void* init_spheroidal() {
	SpheroidalType *spheroidal = (SpheroidalType *) malloc(sizeof(SpheroidalType));
	
	float value = 1.0;
	
	for (int y = 0; y < BLOCKSIZE; y++) {
		for (int x = 0; x < BLOCKSIZE; x++) {
			(*spheroidal)[y][x] = value;
		}
	}

	return spheroidal;
}

void* init_baselines() {
	BaselineType *baselines = (BaselineType *) malloc(sizeof(BaselineType));

	int bl = 0;
	
	for (int station1 = 1 ; station1 < NR_STATIONS; station1++) {
		for (int station2 = 0; station2 < station1; station2++) {
			if (bl >= NR_BASELINES) {
				break;
			}
			(*baselines)[bl].station1 = station1;
			(*baselines)[bl].station2 = station2;
			bl++;
		}
	}
	
	return baselines;
}

void* init_coordinates() {
	UVWType *uvw = (UVWType *) init_uvw();
	CoordinateType *coordinates = (CoordinateType *) malloc(sizeof(CoordinateType));

	for (int bl = 0; bl < NR_BASELINES; bl++) {
		// Get first and last UVW coordinate
		UVW uvw_first = (*uvw)[bl][0];
		UVW uvw_last  = (*uvw)[bl][NR_TIME];

		// Compute average coordinates
		float u_average = ((uvw_first.u + uvw_last.u) / 2);
		float v_average = ((uvw_first.v + uvw_last.v) / 2);
		float w_average = ((uvw_first.w + uvw_last.w) / 2);
	
		// Set position in master grid
		int x = u_average - (BLOCKSIZE / 2);
		int y = v_average - (BLOCKSIZE / 2);
		(*coordinates)[bl] = make_coordinate(x, y);
	}
	
	free(uvw);
	
	return coordinates;
}

void* init_uvgrid() {
	UVGridType *uvgrid = (UVGridType *) malloc(sizeof(UVGridType));
	memset(uvgrid, 0, sizeof(UVGridType));
	return uvgrid;
}

void* init_grid() {
	GridType *grid = (GridType *) malloc(sizeof(GridType));
	memset(grid, 0, sizeof(GridType));
	return grid;
}

}
#else
Init::Init(
    const char *cc, const char *cflags) {
    // Set compile options
    std::stringstream options_init;
	options_init << cflags << " ";
	options_init << SRC_RW << " ";
	options_init << SO_UVW << " ";
	options_init << SO_VIS;

    // Compile util wrapper
	rw::Source(SRC_INIT).compile(cc, SO_INIT, options_init.str().c_str());
	
	// Load module
	module = new rw::Module(SO_INIT);
}

void *Init::init_visibilities() {
    rw::Function(*module, INIT_VISIBILITIES).exec();
}

void *Init::init_uvw() {
    rw::Function(*module, INIT_UVW).exec();
}

void *Init::init_offset() {
    rw::Function(*module, INIT_OFFSET).exec();
}

void *Init::init_wavenumbers() {
    rw::Function(*module, INIT_WAVENUMBERS).exec();
}

void *Init::init_aterm() {
    rw::Function(*module, INIT_ATERM).exec();
}

void *Init::init_spheroidal() {
    rw::Function(*module, INIT_SPHEROIDAL).exec();
}

void *Init::init_baselines() {
    rw::Function(*module, INIT_BASELINES).exec();
}

void *Init::init_coordinates() {
    rw::Function(*module, INIT_COORDINATES).exec();
}

void *Init::init_uvgrid() {
    rw::Function(*module, INIT_UVGRID).exec();
}

void *Init::init_grid() {
    rw::Function(*module, INIT_GRID).exec();
}

int Init::get_nr_stations() {
    return NR_STATIONS;
}

int Init::get_nr_baselines() {
    return NR_BASELINES;
}

int Init::get_nr_baselines_data() {
    return NR_BASELINES_DATA;
}

int Init::get_nr_time() {
    return NR_CHANNELS;
}

int Init::get_nr_time_data() {
    return NR_TIME_DATA;
}

int Init::get_nr_channels() {
    return NR_CHANNELS;
}

int Init::get_nr_polarizations() {
    return NR_POLARIZATIONS;
}

int Init::get_blocksize() {
    return BLOCKSIZE;
}

int Init::get_gridsize() {
    return GRIDSIZE;
}

float Init::get_imagesize() {
    return IMAGESIZE;
}

#endif
