
#include "CompileTimeConstants.h"

using namespace std;

int main(int argc, char *argv[])
{
  idg::ObservationParameters op;

  // this is what has been read from the environment by the defaukt constructor
  op.print();

  // set values explicitly
  op.set_nr_stations(512);
  op.set_nr_timesteps(111);
  op.set_nr_channels(312);
  op.set_nr_polarizations(4); 
  op.set_field_of_view(0.1);

  cout << endl;

  // new values, now using operator<< for output
  cout << op;

  cout << endl;

  // check get functions:
  cout << "get_nr_stations(): " << op.get_nr_stations() << endl;
  cout << "get_nr_timesteps(): " << op.get_nr_timesteps() << endl;
  cout << "get_nr_channels(): " << op.get_nr_channels() << endl;
  cout << "get_nr_polarizations(): " << op.get_nr_polarizations() << endl; 
  cout << "get_field_of_view(): " << op.get_field_of_view() << endl;

  cout << endl;
  

  idg::AlgorithmicParameters ap;

  // this is what has been read from the environment by the defaukt constructor
  ap.print();

  // set values explicitly
  ap.set_grid_size(4096);
  ap.set_subgrid_size(32);
  ap.set_chunk_size(126);
  ap.set_job_size(13);
  ap.set_w_planes(1);

  cout << endl;

  // new values, now using operator<< for output
  cout << ap;

  cout << endl;

  // check get functions:
  cout << "get_grid_size(): " << ap.get_grid_size() << endl;
  cout << "get_subgrid_size(): " << ap.get_subgrid_size() << endl;
  cout << "get_chunk_size(): " << ap.get_chunk_size() << endl;
  cout << "get_job_size(): " << ap.get_job_size() << endl; 
  cout << "get_w_planes(): " << ap.get_w_planes() << endl;

  cout << endl;


  idg::CompileTimeConstants ctc;

  // this is what has been read from the environment by the defaukt constructor
  ctc.print();

  // set values explicitly
  ctc.set_nr_stations(512);
  ctc.set_nr_timesteps(111);
  ctc.set_nr_channels(312);
  ctc.set_nr_polarizations(4); 
  ctc.set_field_of_view(0.1);
  ctc.set_grid_size(4096);
  ctc.set_subgrid_size(32);
  ctc.set_chunk_size(126);
  ctc.set_job_size(13);
  ctc.set_w_planes(1);

  cout << endl;

  // new values, now using operator<< for output
  cout << ctc;
  
  return 0;
}
