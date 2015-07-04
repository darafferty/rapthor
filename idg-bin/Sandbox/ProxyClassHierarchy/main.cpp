
#include "CompileTimeConstants.h"

using namespace std;

int main(int argc, char *argv[])
{
  idg::ObservationParameters op;

  op.print();

  op.set_nr_channels(312);
  op.set_nr_stations(512);
  
  cout << op;


  idg::AlgorithmicParameters ap;

  ap.print();

  ap.set_grid_size(4096);
  ap.set_subgrid_size(32);
  ap.set_chunk_size(126);
  ap.set_job_size(13);

  cout << ap;
  
  return 0;
}
