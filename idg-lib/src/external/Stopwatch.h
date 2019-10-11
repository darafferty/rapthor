#ifndef STOPWATCH_H
#define STOPWATCH_H

#include "idg-config.h"

#include <string>

/**
	@author A.R. Offringa <offringa@astro.rug.nl>
*/
class Stopwatch{
	public:
        static Stopwatch* create();

        virtual ~Stopwatch() {};

		virtual void Start() = 0;
		virtual void Pause() = 0;
		virtual void Reset() = 0;

		virtual std::string ToString() const = 0;
		virtual std::string ToShortString() const = 0;
		virtual long double Seconds() const = 0;

		virtual std::string ToDaysString() const = 0;
		virtual std::string ToHoursString() const = 0;
		virtual std::string ToMinutesString() const = 0;
		virtual std::string ToSecondsString() const = 0;
		virtual std::string ToMilliSecondsString() const = 0;
		virtual std::string ToMicroSecondsString() const = 0;
		virtual std::string ToNanoSecondsString() const = 0;

    protected:
        Stopwatch() {}
};

#endif
