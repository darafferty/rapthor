#include <idg-api.h>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    idg::api::options_type options;
    unique_ptr<idg::api::BufferSet> bufferset (idg::api::BufferSet::create(idg::api::Type::CPU_OPTIMIZED));

    unsigned int buffersize = 4; // Timesteps per buffer
    vector<vector<double>> bands = { {100.e6, 101.e6, 102.e6, 103.e6, 104.e6} };
    int nr_stations = 3;
    float max_baseline = 3000.; // in meters

    unsigned int imagesize = 256;
    float cellsize = 0.01;
    float max_w = 5;

    bufferset->init(imagesize, cellsize, max_w, options);

    bufferset->init_buffers(buffersize,
                            bands,
                            nr_stations,
                            max_baseline,
                            options,
                            idg::api::BufferSetType::gridding);

    size_t subgridsize = bufferset->get_subgridsize();
    cout<<"Subgridsize: "<<subgridsize<<"\n";

    for (size_t timestep=0; timestep<10; ++timestep) {
      for (size_t st1=0; st1<nr_stations; ++st1) {
        for (size_t st2=st1; st2<nr_stations; ++st2) {
          vector<double> uvw = {1., 1., 1.};
          vector<complex<float>> data(4 * bands[0].size(), 0.);
          bufferset->get_gridder(0)->grid_visibilities(timestep,
                                                       st1, st2,
                                                       uvw.data(),
                                                       data.data());
        }
      }
    }
    bufferset->finished();

    return 0;
}
