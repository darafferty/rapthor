#include "PowerRecord.h"

namespace idg {
    namespace kernel {
        namespace opencl {

            void PowerRecord::enqueue(cl::CommandQueue &queue) {
                queue.enqueueMarkerWithWaitList(NULL, &event);
                event.setCallback(CL_RUNNING, &PowerRecord::getPower, this);
            }

            void PowerRecord::getPower(cl_event event, cl_int, void *userData) {
                PowerRecord *record = static_cast<PowerRecord*>(userData);
                record->state = record->sensor->read();
            }

        } // end namespace opencl
    } // end namespace kernel
} // end namespace idg
