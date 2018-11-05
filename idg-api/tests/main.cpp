#include <idg-api.h>
#include <iostream>
#include <math.h>

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

    unsigned int nr_timesteps = 9;
    unsigned int aterm_steps = 3; // Change a-term every 3 time steps
    unsigned int nr_aterms = ceil(nr_timesteps/aterm_steps);

    float shiftl = 0;
    float shiftm = 0;
    float shiftp = 0;

    bufferset->init(imagesize, cellsize, max_w, shiftl, shiftm, shiftp, options);

    bufferset->init_buffers(buffersize,
                            bands,
                            nr_stations,
                            max_baseline,
                            options,
                            idg::api::BufferSetType::gridding);

    size_t subgridsize = bufferset->get_subgridsize();
    cout<<"Subgridsize: "<<subgridsize<<"\n";

    // Initialize a-terms
    vector<complex<float>> aterms(nr_aterms*nr_stations*subgridsize*subgridsize*4, 0.);
    cout<<"Aterms.size() = " <<aterms.size()<<endl;
    for (auto aterm_num=0; aterm_num<nr_aterms; ++aterm_num) {
      for (auto station=0; station<nr_stations; ++station) {
        for (auto x=0; x<subgridsize; ++x) {
          for (auto y=0; x<subgridsize; ++x) {
            size_t offset=4*subgridsize*subgridsize*nr_stations*aterm_num +
                          4*subgridsize*subgridsize*station +
                          4*subgridsize*x + 4*y;
            aterms[offset+0] = 42+aterm_num;
            aterms[offset+1] = 0.;
            aterms[offset+2] = 0.;
            aterms[offset+3] = 42+aterm_num;
          }
        }
      }
    }

    for (size_t timestep=0; timestep<nr_timesteps; ++timestep) {
      if (timestep%aterm_steps==0) {
        size_t atermsize = nr_stations*subgridsize*subgridsize*4;
        bufferset->get_gridder(0)->set_aterm(timestep,
                                             aterms.data()+timestep/aterm_steps*atermsize);
      }

      for (size_t st1=0; st1<nr_stations; ++st1) {
        for (size_t st2=st1; st2<nr_stations; ++st2) {
          vector<double> uvw = {1., 1., 1.};
          vector<complex<float>> data(4 * bands[0].size(), 0.);
          vector<float> weights(data.size(), 1.);
          bufferset->get_gridder(0)->grid_visibilities(timestep,
                                                       st1, st2,
                                                       uvw.data(),
                                                       data.data(),
                                                       weights.data());
        }
      }
    }
    bufferset->finished();

    return 0;
}
