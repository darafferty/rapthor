#include "PowerRecord.h"

namespace idg {
    namespace kernel {
        namespace cuda {

            void PowerRecord::enqueue(cu::Stream &stream) {
                stream.record(event);
                stream.addCallback((CUstreamCallback) &PowerRecord::getPower, &state);
            }

            void PowerRecord::getPower(CUstream, CUresult, void *userData) {
                PowerRecord *record = static_cast<PowerRecord*>(userData);
                record->state = record->sensor->read();
            }

        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg
