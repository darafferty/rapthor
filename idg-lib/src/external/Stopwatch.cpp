#include "StopwatchImpl.h"

#include <cmath>
#include <sstream>

#if defined(HAVE_BOOST)
#include <boost/date_time/posix_time/posix_time.hpp>
#endif

Stopwatch* Stopwatch::create()
{
    return new StopwatchImpl();
}

StopwatchImpl::StopwatchImpl() :
    _running(false)
#if defined(HAVE_BOOST)
    ,
    _sum(boost::posix_time::seconds(0))
#endif
{
}

StopwatchImpl::~StopwatchImpl()
{
}

void StopwatchImpl::Start()
{
	if(!_running) {
#if defined(HAVE_BOOST)
		_startTime = boost::posix_time::microsec_clock::local_time();
#endif
		_running = true;
	}
}

void StopwatchImpl::Pause()
{
	if(_running) {
#if defined(HAVE_BOOST)
		_sum += (boost::posix_time::microsec_clock::local_time() - _startTime);
#endif
		_running = false;
	}
}

void StopwatchImpl::Reset()
{
	_running = false;
#if defined(HAVE_BOOST)
	_sum = boost::posix_time::seconds(0);
#endif
}

std::string StopwatchImpl::ToString() const
{
#if defined(HAVE_BOOST)
	if(_running) {
		boost::posix_time::time_duration current = _sum + (boost::posix_time::microsec_clock::local_time() - _startTime);
		return to_simple_string(current);
	} else {
		return to_simple_string(_sum);
	}
#else
        return std::string("");
#endif
}

std::string StopwatchImpl::ToShortString() const
{
	const long double seconds = Seconds();
	if(seconds >= 60*60*24)
		return ToDaysString();
	else if(seconds >= 60*60)
		return ToHoursString();
	else if(seconds >= 60)
		return ToMinutesString();
	else if(seconds >= 1.0)
		return ToSecondsString();
	else if(seconds >= 0.001)
		return ToMilliSecondsString();
	else if(seconds >= 0.000001)
		return ToMicroSecondsString();
	else
		return ToNanoSecondsString();
}

std::string StopwatchImpl::ToDaysString() const
{
	const long double days = roundl(Seconds() / (60.0*60.0))/24.0;
	std::stringstream str;
	if(days >= 10.0)
		str << roundl(days) << " days";
	else
		str << floorl(days) << 'd' << (days*24.0) << 'h';
	return str.str();
}

std::string StopwatchImpl::ToHoursString() const
{
	const long double hours = roundl(Seconds() / 60.0)/60.0;
	std::stringstream str;
	if(hours >= 10.0)
		str << roundl(hours) << 'h';
	else
		str << floorl(hours) << 'h' << (hours*60.0) << 'm';
	return str.str();
}

std::string StopwatchImpl::ToMinutesString() const
{
	const long double mins = roundl(Seconds())/60.0;
	std::stringstream str;
	if(mins >= 10.0)
		str << roundl(mins) << " min";
	else
		str << floorl(mins) << 'm' << fmod(mins*60.0,60.0) << 's';
	return str.str();
}

std::string StopwatchImpl::ToSecondsString() const
{
	const long double seconds = roundl(Seconds()*10.0)/10.0;
	std::stringstream str;
	if(seconds >= 10.0)
		str << roundl(Seconds()) << 's';
	else
		str << seconds << 's';
	return str.str();
}

std::string StopwatchImpl::ToMilliSecondsString() const
{
	const long double msec = roundl(Seconds()*10000.0)/10.0;
	std::stringstream str;
	if(msec >= 10.0)
		str << roundl(Seconds()*1000.0) << "ms";
	else
		str << msec << "ms";
	return str.str();
}

std::string StopwatchImpl::ToMicroSecondsString() const
{
	const long double usec = roundl(Seconds()*10000000.0)/10.0;
	std::stringstream str;
	if(usec >= 10.0)
		str << roundl(Seconds()*1000000.0) << "µs";
	else
		str << usec << "µs";
	return str.str();
}

std::string StopwatchImpl::ToNanoSecondsString() const
{
	const long double nsec = roundl(Seconds()*10000000000.0)/10.0;
	std::stringstream str;
	if(nsec >= 10.0)
		str << roundl(Seconds()*1000000000.0) << "ns";
	else
		str << nsec << "ns";
	return str.str();
}

long double StopwatchImpl::Seconds() const
{
#if defined(HAVE_BOOST)
	if(_running) {
		boost::posix_time::time_duration current = _sum + (boost::posix_time::microsec_clock::local_time() - _startTime);
		return (long double) current.total_microseconds()/1000000.0;
	} else {
		return (long double) _sum.total_microseconds()/1000000.0;
	}
#else
    return 0;
#endif
}
