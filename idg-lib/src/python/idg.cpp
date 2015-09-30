#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/python/extract.hpp>

#include <AbstractProxy.h>
#include <CPU/common/idg-cpu.h>

#include <iostream>

#include <numpy/arrayobject.h>
 
using namespace boost::python;
using namespace idg;
using namespace idg::proxy;

// Functions to demonstrate extraction
void setArray(boost::python::numeric::array data) {
 
    // Access a built-in type (an array)
    // Need to <extract> array elements because their type is unknown
//     std::cout << "First array item: " << extract<int>(a[0][0]) << std::endl;

    std::cout << "pointer: " << data.ptr() << std::endl;
    
    int *p = (int*)PyArray_DATA(data.ptr());
    std::cout << "pointer: " << p << std::endl;
    std::cout << "value: " << *p << std::endl;
    
    int i = extract<int>(data.attr("nbytes"));
    std::cout << "nbytes: " << i << std::endl;
//     std::cout << extract<int*>(data.attr("data"));
    
    PyArrayObject* pao = (PyArrayObject*)(data.ptr());
    std::cout << pao->nd << std::endl;
    for(int j = 0; j<pao->nd; j++)
    {
      std::cout << "  " << pao->dimensions[j] << std::endl;
    }
    
    std::cout << pao->descr << std::endl;
    std::cout << pao->descr->kind << std::endl;
    std::cout << pao->descr->elsize << std::endl;
    
      //       PyArray_FROM_O(b);
    
//     std::cout << "pointer: " << p << std::endl;
 //     std::cout << "value: " << *p << std::endl;
    
}
 
 
void grid_onto_subgrids(
  Proxy *p, 
  int jobsize, 
  int nr_subgrids, 
  float w_offset, 
  boost::python::numeric::array uvw, 
  boost::python::numeric::array wavenumbers, 
  boost::python::numeric::array visibilities,
  boost::python::numeric::array spheroidal,
  boost::python::numeric::array aterm,
  boost::python::numeric::array metadata,
  boost::python::numeric::array subgrids
)
{
  // TODO: check type and shape of arrays
  p->grid_onto_subgrids(
    jobsize,
    nr_subgrids,
    w_offset,
    ((PyArrayObject*)(uvw.ptr()))->data,
    ((PyArrayObject*)(wavenumbers.ptr()))->data,
    ((PyArrayObject*)(visibilities.ptr()))->data,
    ((PyArrayObject*)(spheroidal.ptr()))->data,
    ((PyArrayObject*)(aterm.ptr()))->data,
    ((PyArrayObject*)(metadata.ptr()))->data,
    ((PyArrayObject*)(subgrids.ptr()))->data
  );
}


void add_subgrids_to_grid(
  Proxy *p, 
  int jobsize, 
  unsigned nr_subgrids, 
  boost::python::numeric::array metadata, 
  boost::python::numeric::array subgrids, 
  boost::python::numeric::array grid
)
{
  // TODO: check type and shape of arrays
  p->add_subgrids_to_grid(
    jobsize,
    nr_subgrids,
    ((PyArrayObject*)(metadata.ptr()))->data,
    ((PyArrayObject*)(subgrids.ptr()))->data,
    ((PyArrayObject*)(grid.ptr()))->data
  );
}

BOOST_PYTHON_MODULE(_idg)
{
  
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
 
  def("setArray", &setArray);

  void (Parameters::*print0)() const = &Parameters::print;
  
  class_<Parameters>("Parameters")
  .add_property("nr_stations", &Parameters::get_nr_stations, &Parameters::set_nr_stations)
  .add_property("nr_baselines", &Parameters::get_nr_baselines)
  .add_property("nr_channels", &Parameters::get_nr_channels, &Parameters::set_nr_channels)
  .add_property("nr_timesteps", &Parameters::get_nr_timesteps, &Parameters::set_nr_timesteps)
  .add_property("nr_timeslots", &Parameters::get_nr_timeslots, &Parameters::set_nr_timeslots)
  .add_property("imagesize", &Parameters::get_imagesize, &Parameters::set_imagesize)
  .add_property("grid_size", &Parameters::get_grid_size, &Parameters::set_grid_size)
  .add_property("subgrid_size", &Parameters::get_subgrid_size, &Parameters::set_subgrid_size)
  .add_property("job_size", &Parameters::get_job_size, &Parameters::set_job_size)
  .add_property("job_size_gridding", &Parameters::get_job_size_gridding, &Parameters::set_job_size_gridding)
  .add_property("job_size_degridding", &Parameters::get_job_size_degridding, &Parameters::set_job_size_degridding)
  .add_property("job_size_gridder", &Parameters::get_job_size_gridder, &Parameters::set_job_size_gridder)
  .add_property("job_size_adder", &Parameters::get_job_size_adder, &Parameters::set_job_size_adder)
  .add_property("job_size_splitter", &Parameters::get_job_size_splitter, &Parameters::set_job_size_splitter)
  .add_property("job_size_degridder", &Parameters::get_job_size_degridder, &Parameters::set_job_size_degridder)
  .add_property("nr_polarizations", &Parameters::get_nr_polarizations)
  .def("print0", print0);
  
  class_<Proxy, boost::noncopyable>("Proxy", no_init)
  .def("grid_onto_subgrids", &grid_onto_subgrids)
  .def("add_subgrids_to_grid", &add_subgrids_to_grid)
  
  ;
  
  class_<proxy::CPU, bases<proxy::Proxy>>("CPU", init<Parameters>());
}



// typedef struct { float u, v, w; } UVW;
// typedef struct { int x, y; } Coordinate;
// typedef struct { int station1, station2; } Baseline;
// typedef struct { int time_nr; Baseline baseline; Coordinate coordinate; } Metadata;
// 
// /*
//     Complex numbers
// */
// #define FLOAT_COMPLEX std::complex<float>
// 
// /*
//     Datatypes
// */
// typedef UVW UVWType[1][NR_TIMESTEPS];
// typedef FLOAT_COMPLEX VisibilitiesType[1][NR_TIMESTEPS][NR_CHANNELS][NR_POLARIZATIONS];
// typedef float WavenumberType[NR_CHANNELS];
// typedef FLOAT_COMPLEX ATermType[NR_STATIONS][NR_TIMESLOTS][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
// typedef float SpheroidalType[SUBGRIDSIZE][SUBGRIDSIZE];
// typedef FLOAT_COMPLEX GridType[NR_POLARIZATIONS][GRIDSIZE][GRIDSIZE];
// typedef FLOAT_COMPLEX SubGridType[1][NR_POLARIZATIONS][SUBGRIDSIZE][SUBGRIDSIZE];
// typedef Metadata MetadataType[1];
