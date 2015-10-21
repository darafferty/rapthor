#include <iostream>

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/python/extract.hpp>

#include <AbstractProxy.h>
#include <CPU/HaswellEP/idg.h>

#include <numpy/arrayobject.h>


using namespace boost::python;
using namespace idg;
using namespace idg::proxy;


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

void transform(
  Proxy *p,
  unsigned _direction,
  boost::python::numeric::array grid
)
{
    DomainAtoDomainB direction = _direction ?
        FourierDomainToImageDomain : ImageDomainToFourierDomain;
  p->transform(
    direction,
    ((PyArrayObject*)(grid.ptr()))->data
  );
}

void split_grid_into_subgrids(
  Proxy *p,
  int jobsize,
  unsigned nr_subgrids,
  boost::python::numeric::array metadata,
  boost::python::numeric::array subgrids,
  boost::python::numeric::array grid
)
{
  // TODO: check type and shape of arrays
  p->split_grid_into_subgrids(
    jobsize,
    nr_subgrids,
    ((PyArrayObject*)(metadata.ptr()))->data,
    ((PyArrayObject*)(subgrids.ptr()))->data,
    ((PyArrayObject*)(grid.ptr()))->data
  );
}

void degrid_from_subgrids(
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
  p->degrid_from_subgrids(
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

BOOST_PYTHON_MODULE(_idg)
{
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

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
  .def("transform", &transform)
  .def("split_grid_into_subgrids", &split_grid_into_subgrids)
  .def("degrid_from_subgrids", &degrid_from_subgrids)

  ;

  class_<proxy::cpu::HaswellEP, bases<proxy::Proxy>>("HaswellEP", init<Parameters>());
}
