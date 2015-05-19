#include <stdint.h>
#include <complex.h>
#include <math.h>

#include <idg/Common/RW.h>

class KernelGridder {
    public:
        KernelGridder(rw::Module &module, const char *kernel);
        void run(
            int jobsize,
            void *uvw, void *offset,
            void *wavenumbers, void *visibilities,
            void *spheroidal, void *aterm,
            void *baselines, void *uvgrid);
    	uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
	
	private:
	    rw::Function function;
};

class KernelDegridder {
    public:
        KernelDegridder(rw::Module &module, const char *kernel);
        void run(
            int jobsize,
            void *uvgrid, void *uvw,
            void *offset, void *wavenumbers,
            void *aterm, void *baselines,
            void *spheroidal, void *visibilities);
    	uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
	
	private:
	    rw::Function function;
};

class KernelFFT {
	public:
		KernelFFT(rw::Module &module, const char *kernel);
		void run(int size, int batch, void *data, int direction);
		uint64_t flops(int size, int batch);
		uint64_t bytes(int size, int batch);
		
	private:
		rw::Function function;
};

class KernelAdder {
	public:
		KernelAdder(rw::Module &module, const char *kernel);
		void run(
			int jobsize,
			void *coordinates,
			void *uvgrid,
			void *grid);
		uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
		
	private:
		rw::Function function;
};

class KernelSplitter {
	public:
		KernelSplitter(rw::Module &module, const char *kernel);
		void run(
			int jobsize,
			void *coordinates,
			void *uvgrid,
			void *grid);
		uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
		
	private:
		rw::Function function;
};

class KernelShifter {
	public:
		KernelShifter(rw::Module &module, const char *kernel);
		void run(
			int jobsize,
			void *uvgrid);
		uint64_t flops(int jobsize);
		uint64_t bytes(int jobsize);
		
	private:
		rw::Function function;
};
