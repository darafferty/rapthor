#include "PowerRecord.h"

namespace idg {
    namespace kernel {
        namespace cuda {

            PowerRecord::PowerRecord() {};

            PowerRecord::PowerRecord(
                powersensor::PowerSensor *sensor) :
                sensor(sensor) {}

            void PowerRecord::enqueue(cu::Stream &stream) {
                stream.record(event);
                stream.addCallback((CUstreamCallback) &PowerRecord::getPower, this);
            }

            void PowerRecord::getPower(CUstream, CUresult, void *userData) {
                PowerRecord *record = static_cast<PowerRecord*>(userData);
                record->state = record->sensor->read();
            }

        } // end namespace cuda
    } // end namespace kernel
} // end namespace idg
