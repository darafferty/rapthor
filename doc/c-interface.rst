C Interface
===========

The C-interface provides access to the C++ classes through function calls.
Because C does not have classes they are passed as opaque pointers, i.e.
pointers to an incomplete type.

A proxy instance can be created by a call to one of the constructor functions:

- :cpp:func:`CPU_Optimized_create` from :ref:`CPU-OptimizedC.h` to create an instance of :cpp:class:`idg::proxy::cpu::Optimized`.
- :cpp:func:`CPU_Reference_create` from :ref:`CPU-ReferenceC.h` to create an instance of :cpp:class:`idg::proxy::cpu::Reference`.
- :cpp:func:`CUDA_Generic_create` from :ref:`CUDA-GenericC.h` to create an instance of :cpp:class:`idg::proxy::cuda::Generic`.
- :cpp:func:`HybridCUDA_GenericOptimized_create` from :ref:`Hybrid-GenericOptimizedC.h` to create an instance of :cpp:class:`idg::proxy::hybrid::GenericOptimized`.

Afterwards the proxy need to be destroyed by a call to :cpp:func:`Proxy_destroy`

.. toctree::
   :maxdepth: 2

   ProxyC.h.rst
   CPU-ReferenceC.h.rst
   CPU-OptimizedC.h.rst
   CUDA-GenericC.h.rst
   Hybrid-GenericOptimizedC.h.rst

