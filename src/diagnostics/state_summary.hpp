#ifndef DIAGNOSTICS_STATE_SUMMARY_HPP_
#define DIAGNOSTICS_STATE_SUMMARY_HPP_

// Host-only, read-only summaries for the optional end-of-run comparison.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ostream>

namespace diagnostics {
struct StateSummary {
  std::uint64_t count=0, finite=0, nan=0, posinf=0, neginf=0;
  double minimum=0.0, maximum=0.0, scale=0.0;
  long double sum=0.0L, absolute=0.0L, squares=0.0L;

  void Add(double value) {
    ++count;
    if (!std::isfinite(value)) {
      if (std::isnan(value)) ++nan;
      else if (value>0) ++posinf;
      else ++neginf;
      return;
    }
    if (finite==0) minimum=maximum=value;
    else {minimum=std::min(minimum,value); maximum=std::max(maximum,value);}
    ++finite;
    const double magnitude=std::fabs(value);
    if (magnitude>scale) {
      const long double ratio=scale/magnitude;
      sum*=ratio; absolute*=ratio; squares*=ratio*ratio;
      scale=magnitude;
    }
    const long double normalized=scale>0.0 ? value/scale : 0.0;
    sum+=normalized; absolute+=std::fabs(normalized); squares+=normalized*normalized;
  }

  void Write(std::ostream &out) const {
    // Scaled accumulation avoids overflow in sums and squared norms. Clamp only
    // rounding overshoots in normalized averages, not any physical state value.
    const long double n=finite ? finite : 1;
    const double mean=scale*static_cast<double>(std::max(-1.0L,std::min(1.0L,sum/n)));
    const double l1=scale*static_cast<double>(std::min(1.0L,absolute/n));
    const double rms=scale*static_cast<double>(std::sqrt(std::min(1.0L,squares/n)));
    out<<"{\"count\":"<<count<<",\"finite\":"<<finite<<",\"nan\":"<<nan
       <<",\"posinf\":"<<posinf<<",\"neginf\":"<<neginf
       <<",\"mean\":"<<mean<<",\"mean_abs\":"<<l1<<",\"rms\":"<<rms
       <<",\"min\":"<<minimum<<",\"max\":"<<maximum<<"}";
  }
};
}  // namespace diagnostics
#endif
