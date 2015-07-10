
#include "Parameters.h"

using namespace std;

int main(int argc, char *argv[])
{
  idg::Parameters p;

  // this is what has been read from the environment by the defaukt constructor
  p.print();

  // set values explicitly
  p.set_nr_stations(512);
  p.set_nr_timesteps(111);
  p.set_nr_channels(312);
  p.set_nr_polarizations(4); 
  p.set_field_of_view(0.1);
  p.set_grid_size(4096);
  p.set_w_planes(1);

  cout << endl;

  // new values, now using operator<< for output
  cout << p;
  
  return 0;
}
