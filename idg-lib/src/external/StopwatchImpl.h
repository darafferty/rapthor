#include "Stopwatch.h"

#include <string>

#if defined(HAVE_BOOST)
#include <boost/date_time/posix_time/posix_time_types.hpp>
#endif

class StopwatchImpl : public virtual Stopwatch {

    public:
        StopwatchImpl();

        ~StopwatchImpl();

        virtual void Start() final override;
		virtual void Pause() final override;
		virtual void Reset() final override;

		virtual std::string ToString() const final override;
		virtual std::string ToShortString() const final override;
		virtual long double Seconds() const final override;

		virtual std::string ToDaysString() const final override;
		virtual std::string ToHoursString() const final override;
		virtual std::string ToMinutesString() const final override;
		virtual std::string ToSecondsString() const final override;
		virtual std::string ToMilliSecondsString() const final override;
		virtual std::string ToMicroSecondsString() const final override;
		virtual std::string ToNanoSecondsString() const final override;

    private:
		bool _running;
        #if defined(HAVE_BOOST)
		boost::posix_time::ptime _startTime;
		boost::posix_time::time_duration _sum;
        #endif
};
