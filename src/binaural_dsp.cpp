#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

const double PI = 3.14159265358979323846;

// ─────────────────────────────────────────────────────────────────────────────
//  Biquad IIR Filter (Direct Form II Transposed)
// ─────────────────────────────────────────────────────────────────────────────
class BiquadFilter {
public:
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0; // a0 is normalized to 1.0
    double s1 = 0.0, s2 = 0.0;

    void reset() {
        s1 = 0.0;
        s2 = 0.0;
    }

    void set_lpf(double frequency, double sample_rate, double q = 0.707) {
        if (frequency >= sample_rate * 0.49) {
            // Bypass
            b0 = 1.0; b1 = 0.0; b2 = 0.0;
            a1 = 0.0; a2 = 0.0;
            return;
        }
        double w0 = 2.0 * PI * frequency / sample_rate;
        double alpha = std::sin(w0) / (2.0 * q);
        double cosw0 = std::cos(w0);

        double a0 = 1.0 + alpha;
        b0 = ((1.0 - cosw0) / 2.0) / a0;
        b1 = (1.0 - cosw0) / a0;
        b2 = ((1.0 - cosw0) / 2.0) / a0;
        a1 = (-2.0 * cosw0) / a0;
        a2 = (1.0 - alpha) / a0;
    }

    void set_high_shelf(double frequency, double sample_rate, double gain_db, double q = 0.707) {
        double A = std::pow(10.0, gain_db / 40.0);
        double w0 = 2.0 * PI * frequency / sample_rate;
        double alpha = std::sin(w0) / (2.0 * q);
        double cosw0 = std::cos(w0);
        double two_sqrt_A_alpha = 2.0 * std::sqrt(A) * alpha;

        double a0 = (A + 1.0) - (A - 1.0) * cosw0 + two_sqrt_A_alpha;
        b0 = (A * ((A + 1.0) + (A - 1.0) * cosw0 + two_sqrt_A_alpha)) / a0;
        b1 = (-2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0)) / a0;
        b2 = (A * ((A + 1.0) + (A - 1.0) * cosw0 - two_sqrt_A_alpha)) / a0;
        a1 = (2.0 * ((A - 1.0) - (A + 1.0) * cosw0)) / a0;
        a2 = ((A + 1.0) - (A - 1.0) * cosw0 - two_sqrt_A_alpha) / a0;
    }

    inline float process(float x) {
        double y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        return static_cast<float>(y);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Circular Delay Line (Integer delay for ITD)
// ─────────────────────────────────────────────────────────────────────────────
class DelayLine {
private:
    std::vector<float> buffer;
    int write_index = 0;
    int delay_samples = 0;

public:
    DelayLine(int max_delay = 1024) {
        buffer.assign(max_delay, 0.0f);
    }

    void set_delay(double samples) {
        delay_samples = std::clamp(static_cast<int>(std::round(samples)), 0, static_cast<int>(buffer.size() - 1));
    }

    inline float process(float x) {
        buffer[write_index] = x;
        int read_index = write_index - delay_samples;
        if (read_index < 0) {
            read_index += buffer.size();
        }
        write_index = (write_index + 1) % buffer.size();
        return buffer[read_index];
    }

    void reset() {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        write_index = 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Binaural Processor
// ─────────────────────────────────────────────────────────────────────────────
class BinauralProcessor {
private:
    double sample_rate;
    double head_radius = 0.0875; // meters
    double speed_of_sound = 343.0; // m/s

    // Position parameters
    double current_azimuth = 0.0;
    double current_elevation = 0.0;
    double current_distance = 1.0;

    // Spatial processing states
    DelayLine delay_L;
    DelayLine delay_R;
    
    BiquadFilter dist_lpf_L;
    BiquadFilter dist_lpf_R;
    
    BiquadFilter elevation_shelf_L;
    BiquadFilter elevation_shelf_R;
    
    BiquadFilter ild_shelf_L;
    BiquadFilter ild_shelf_R;
    
    BiquadFilter rear_shelf_L;
    BiquadFilter rear_shelf_R;

    double distance_gain = 1.0;
    double azimuth_gain_L = 1.0;
    double azimuth_gain_R = 1.0;

public:
    BinauralProcessor(double sr) : sample_rate(sr), delay_L(256), delay_R(256) {
        reset();
    }

    void reset() {
        delay_L.reset();
        delay_R.reset();
        dist_lpf_L.reset();
        dist_lpf_R.reset();
        elevation_shelf_L.reset();
        elevation_shelf_R.reset();
        ild_shelf_L.reset();
        ild_shelf_R.reset();
        rear_shelf_L.reset();
        rear_shelf_R.reset();
        set_position(0.0, 0.0, 1.0);
    }

    void set_position(double azimuth_deg, double elevation_deg, double distance_m) {
        current_azimuth = azimuth_deg;
        current_elevation = elevation_deg;
        current_distance = std::max(0.1, distance_m);

        double az_rad = azimuth_deg * PI / 180.0;
        double el_rad = elevation_deg * PI / 180.0;

        // 1. Distance Gain & Air Absorption LPF
        distance_gain = 1.0 / current_distance;
        // Frequency cutoff decreases with distance to simulate air absorption
        double lpf_freq = 20000.0 / (1.0 + 0.15 * (current_distance - 1.0));
        lpf_freq = std::clamp(lpf_freq, 1000.0, 20000.0);
        dist_lpf_L.set_lpf(lpf_freq, sample_rate);
        dist_lpf_R.set_lpf(lpf_freq, sample_rate);

        // 2. ITD (Inter-aural Time Delay)
        // Woodworth's formula: Delay = (r/c) * (theta + sin(theta))
        // If sound comes from the right (azimuth > 0), the left ear is delayed.
        double delay_seconds = 0.0;
        double abs_az_rad = std::abs(az_rad);
        if (abs_az_rad <= PI / 2.0) {
            delay_seconds = (head_radius / speed_of_sound) * (abs_az_rad + std::sin(abs_az_rad));
        } else {
            double theta = PI - abs_az_rad;
            delay_seconds = (head_radius / speed_of_sound) * (theta + std::sin(theta));
        }
        double delay_samples = delay_seconds * sample_rate;

        if (az_rad >= 0.0) {
            // Source is on the right -> delay left ear
            delay_L.set_delay(delay_samples);
            delay_R.set_delay(0.0);
        } else {
            // Source is on the left -> delay right ear
            delay_L.set_delay(0.0);
            delay_R.set_delay(delay_samples);
        }

        // 3. ILD (Inter-aural Level Difference)
        // High-shelf filter at 2000 Hz. Near ear gets boost, far ear gets cut.
        double max_ild_db = 6.0;
        double ild_db = max_ild_db * std::sin(az_rad); // positive if source is on right
        
        ild_shelf_R.set_high_shelf(2000.0, sample_rate, ild_db);      // boosted if ild_db > 0 (right)
        ild_shelf_L.set_high_shelf(2000.0, sample_rate, -ild_db);     // cut if ild_db > 0

        // Equal-power panning for overall level balance
        // We map azimuth to effective front-hemisphere azimuth for left/right volume balance,
        // and let the rear filter handle the front/back cues.
        double eff_az_deg = azimuth_deg;
        if (azimuth_deg > 90.0) {
            eff_az_deg = 180.0 - azimuth_deg;
        } else if (azimuth_deg < -90.0) {
            eff_az_deg = -180.0 - azimuth_deg;
        }

        // Map effective azimuth [-90, 90] to alpha [0, 1]
        double alpha = (eff_az_deg + 90.0) / 180.0;
        azimuth_gain_L = std::cos(alpha * PI / 2.0);
        azimuth_gain_R = std::sin(alpha * PI / 2.0);


        // 4. Elevation Pinna Cues
        // High-shelf filter at 8000 Hz. Boost when source is above, cut when below.
        double elevation_gain_db = 6.0 * std::sin(el_rad);
        elevation_shelf_L.set_high_shelf(8000.0, sample_rate, elevation_gain_db);
        elevation_shelf_R.set_high_shelf(8000.0, sample_rate, elevation_gain_db);

        // 5. Rear Hemisphere HF Cut
        // If sound source is behind us (|azimuth| > 90), apply a high-shelf cut at 4000 Hz
        if (abs_az_rad > PI / 2.0) {
            double angle_past_90 = abs_az_rad - PI / 2.0; // 0 to PI/2
            double rear_cut_db = -8.0 * std::sin(angle_past_90);
            rear_shelf_L.set_high_shelf(4000.0, sample_rate, rear_cut_db);
            rear_shelf_R.set_high_shelf(4000.0, sample_rate, rear_cut_db);
        } else {
            rear_shelf_L.set_high_shelf(4000.0, sample_rate, 0.0);
            rear_shelf_R.set_high_shelf(4000.0, sample_rate, 0.0);
        }
    }

    void process(const float* input, float* output_left, float* output_right, int frames) {
        for (int i = 0; i < frames; ++i) {
            float in_sample = input[i];

            // 1. Distance attenuation & LPF
            float mono_dist_L = dist_lpf_L.process(in_sample) * static_cast<float>(distance_gain);
            float mono_dist_R = dist_lpf_R.process(in_sample) * static_cast<float>(distance_gain);

            // 2. Rear HF cut
            float mono_rear_L = rear_shelf_L.process(mono_dist_L);
            float mono_rear_R = rear_shelf_R.process(mono_dist_R);

            // 3. Elevation Cues
            float mono_el_L = elevation_shelf_L.process(mono_rear_L);
            float mono_el_R = elevation_shelf_R.process(mono_rear_R);

            // 4. ITD Delays
            float delayed_L = delay_L.process(mono_el_L);
            float delayed_R = delay_R.process(mono_el_R);

            // 5. ILD filtering
            float ild_L = ild_shelf_L.process(delayed_L);
            float ild_R = ild_shelf_R.process(delayed_R);

            // 6. Azimuth panning scale
            output_left[i] = ild_L * static_cast<float>(azimuth_gain_L);
            output_right[i] = ild_R * static_cast<float>(azimuth_gain_R);
        }
    }

    // Pybind11 wrapper for NumPy arrays
    std::pair<py::array_t<float>, py::array_t<float>> process_py(py::array_t<float> input_array) {
        py::buffer_info buf = input_array.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Input array must be 1-dimensional");
        }
        int size = buf.shape[0];
        const float* input_ptr = static_cast<const float*>(buf.ptr);

        // Prepare output arrays
        auto output_left = py::array_t<float>(size);
        auto output_right = py::array_t<float>(size);
        
        float* out_L_ptr = static_cast<float*>(output_left.request().ptr);
        float* out_R_ptr = static_cast<float*>(output_right.request().ptr);

        process(input_ptr, out_L_ptr, out_R_ptr, size);

        return {output_left, output_right};
    }
};

PYBIND11_MODULE(binaural_dsp, m) {
    m.doc() = "C++ High Performance Binaural 3D Spatial DSP Engine";
    py::class_<BinauralProcessor>(m, "BinauralProcessor")
        .def(py::init<double>())
        .def("reset", &BinauralProcessor::reset)
        .def("set_position", &BinauralProcessor::set_position)
        .def("process", &BinauralProcessor::process_py);
}
