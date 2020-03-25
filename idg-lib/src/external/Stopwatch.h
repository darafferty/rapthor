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
		virtual long double Seconds() const = 0;

    protected:
        Stopwatch() {}
};

#endif
