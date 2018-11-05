#include "PowerRecord.h"

#include <csignal>

namespace idg {
    namespace kernel {
        namespace opencl {

            PowerRecord::PowerRecord() {};

            PowerRecord::PowerRecord(
                powersensor::PowerSensor *sensor) :
                sensor(sensor) {}

            void signal_handler(int sig)
            {
                // Ignore signal
            }

            void PowerRecord::enqueue(cl::CommandQueue &queue) {
                // Hack to ignore signals that might occur
                // when using the Nvidia OpenCL runtime
                struct sigaction act;
                act.sa_handler = signal_handler;
                sigemptyset(&act.sa_mask);
                act.sa_flags = 0;
                sigaction(SIGSEGV, &act, 0);
                sigaction(SIGILL, &act, 0);

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
