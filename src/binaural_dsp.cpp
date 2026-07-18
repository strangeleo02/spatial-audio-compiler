#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

const double PI = 3.14159265358979323846;

// Standard CIPIC azimuth coordinate grid values
const double cipic_az[25] = {
    -80.0, -65.0, -55.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 
    0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 55.0, 65.0, 80.0
};

// ─────────────────────────────────────────────────────────────────────────────
//  Biquad IIR Filter (Direct Form II Transposed) — Used for distance LPF
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

    inline float process(float x) {
        double y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        return static_cast<float>(y);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Binaural Processor using Measured KEMAR HRTF Convolution
// ─────────────────────────────────────────────────────────────────────────────
class BinauralProcessor {
private:
    double sample_rate;
    int hrir_len = 0;

    // KEMAR HRTF Database (CIPIC Subject 021)
    // Arrays have shape (25, 50, hrir_len)
    std::vector<double> db_hrir_l;
    std::vector<double> db_hrir_r;
    int num_az = 25;
    int num_el = 50;

    // Position parameters
    double current_azimuth = 0.0;
    double current_elevation = 0.0;
    double current_distance = 1.0;
    double distance_gain = 1.0;

    // DSP state
    std::vector<float> history;
    int write_idx = 0;

    // Active & target coefficients
    std::vector<float> h_l_new;
    std::vector<float> h_r_new;
    std::vector<float> h_l_old;
    std::vector<float> h_r_old;

    bool position_changed = false;

    // Distance LPF
    BiquadFilter dist_lpf;

    inline double get_hrir_val(bool is_left, int az_idx, int el_idx, int n) const {
        int idx = (az_idx * num_el + el_idx) * hrir_len + n;
        return is_left ? db_hrir_l[idx] : db_hrir_r[idx];
    }

public:
    BinauralProcessor(double sr) : sample_rate(sr) {
        reset();
    }

    void reset() {
        std::fill(history.begin(), history.end(), 0.0f);
        write_idx = 0;
        position_changed = false;
        dist_lpf.reset();
        if (hrir_len > 0) {
            set_position(0.0, 0.0, 1.0);
            std::copy(h_l_new.begin(), h_l_new.end(), h_l_old.begin());
            std::copy(h_r_new.begin(), h_r_new.end(), h_r_old.begin());
        }
    }

    void load_hrir_database(py::array_t<double> hrir_l, py::array_t<double> hrir_r) {
        py::buffer_info buf_l = hrir_l.request();
        py::buffer_info buf_r = hrir_r.request();

        if (buf_l.ndim != 3 || buf_r.ndim != 3) {
            throw std::runtime_error("HRIR arrays must be 3-dimensional (25, 50, L)");
        }

        num_az = buf_l.shape[0];
        num_el = buf_l.shape[1];
        hrir_len = buf_l.shape[2];

        // Allocate database memory
        int total_size = num_az * num_el * hrir_len;
        db_hrir_l.assign(static_cast<double*>(buf_l.ptr), static_cast<double*>(buf_l.ptr) + total_size);
        db_hrir_r.assign(static_cast<double*>(buf_r.ptr), static_cast<double*>(buf_r.ptr) + total_size);

        // Resize DSP state buffers
        history.assign(hrir_len, 0.0f);
        write_idx = 0;

        h_l_new.assign(hrir_len, 0.0f);
        h_r_new.assign(hrir_len, 0.0f);
        h_l_old.assign(hrir_len, 0.0f);
        h_r_old.assign(hrir_len, 0.0f);

        position_changed = false;

        // Force initial position computation
        set_position(current_azimuth, current_elevation, current_distance);
        
        // Copy initial target coefficients to old coefficients
        std::copy(h_l_new.begin(), h_l_new.end(), h_l_old.begin());
        std::copy(h_r_new.begin(), h_r_new.end(), h_r_old.begin());
    }

    void set_position(double azimuth_deg, double elevation_deg, double distance_m) {
        current_azimuth = azimuth_deg;
        current_elevation = elevation_deg;
        current_distance = std::max(0.1, distance_m);

        // 1. Distance attenuation & LPF cutoff
        distance_gain = 1.0 / current_distance;
        double lpf_freq = 20000.0 / (1.0 + 0.15 * (current_distance - 1.0));
        lpf_freq = std::clamp(lpf_freq, 1000.0, 20000.0);
        dist_lpf.set_lpf(lpf_freq, sample_rate);

        // If the database is not loaded yet, stop here.
        if (hrir_len <= 0) return;

        // Convert angles to radians
        double az_rad = azimuth_deg * PI / 180.0;
        double el_rad = elevation_deg * PI / 180.0;

        // 2. Map standard spherical coordinates to Interaural-Polar coordinates
        // theta_ip = arcsin(cos(el) * sin(az))
        double theta_ip_rad = std::asin(std::cos(el_rad) * std::sin(az_rad));
        double theta_ip_deg = theta_ip_rad * 180.0 / PI;

        // phi_ip = atan2(sin(el), cos(el) * cos(az))
        double phi_ip_rad = std::atan2(std::sin(el_rad), std::cos(el_rad) * std::cos(az_rad));
        double phi_ip_deg = phi_ip_rad * 180.0 / PI;

        // Clamp azimuth to CIPIC range [-80.0, 80.0]
        double az_val = std::clamp(theta_ip_deg, -80.0, 80.0);

        // Map elevation to CIPIC range [-45.0, 230.625]
        double el_val = phi_ip_deg;
        if (el_val < -45.0) {
            el_val += 360.0;
        }
        el_val = std::clamp(el_val, -45.0, 230.625);

        // 3. Find bilinear interpolation indices & weights
        // Elevation (uniformly spaced every 5.625 degrees, starting at -45.0)
        double el_idx_float = (el_val - (-45.0)) / 5.625;
        int el_idx_1 = std::clamp(static_cast<int>(std::floor(el_idx_float)), 0, num_el - 1);
        int el_idx_2 = std::clamp(el_idx_1 + 1, 0, num_el - 1);
        double el_frac = el_idx_float - el_idx_1;

        // Azimuth (non-uniformly spaced)
        auto it = std::lower_bound(cipic_az, cipic_az + 25, az_val);
        int az_idx_2 = std::distance(cipic_az, it);
        if (az_idx_2 == 0) {
            az_idx_2 = 1;
        } else if (az_idx_2 == 25) {
            az_idx_2 = 24;
        }
        int az_idx_1 = az_idx_2 - 1;
        double az_val_1 = cipic_az[az_idx_1];
        double az_val_2 = cipic_az[az_idx_2];
        double az_frac = (az_val - az_val_1) / (az_val_2 - az_val_1);

        // Bilinear interpolation weights
        double w11 = (1.0 - az_frac) * (1.0 - el_frac);
        double w21 = az_frac * (1.0 - el_frac);
        double w12 = (1.0 - az_frac) * el_frac;
        double w22 = az_frac * el_frac;

        // 4. Interpolate HRIR left and right filters
        // Copy current target to old coefficients
        std::copy(h_l_new.begin(), h_l_new.end(), h_l_old.begin());
        std::copy(h_r_new.begin(), h_r_new.end(), h_r_old.begin());

        for (int n = 0; n < hrir_len; ++n) {
            double val_l = w11 * get_hrir_val(true, az_idx_1, el_idx_1, n) +
                           w21 * get_hrir_val(true, az_idx_2, el_idx_1, n) +
                           w12 * get_hrir_val(true, az_idx_1, el_idx_2, n) +
                           w22 * get_hrir_val(true, az_idx_2, el_idx_2, n);

            double val_r = w11 * get_hrir_val(false, az_idx_1, el_idx_1, n) +
                           w21 * get_hrir_val(false, az_idx_2, el_idx_1, n) +
                           w12 * get_hrir_val(false, az_idx_1, el_idx_2, n) +
                           w22 * get_hrir_val(false, az_idx_2, el_idx_2, n);

            h_l_new[n] = static_cast<float>(val_l);
            h_r_new[n] = static_cast<float>(val_r);
        }

        position_changed = true;
    }

    void process(const float* input, float* output_left, float* output_right, int frames) {
        if (hrir_len <= 0) {
            // Bypass if database not loaded
            std::copy(input, input + frames, output_left);
            std::copy(input, input + frames, output_right);
            return;
        }

        for (int i = 0; i < frames; ++i) {
            float in_sample = input[i];

            // 1. Distance attenuation & LPF
            float filtered_mono = dist_lpf.process(in_sample) * static_cast<float>(distance_gain);
            
            // Write to history
            history[write_idx] = filtered_mono;

            float sum_l = 0.0f;
            float sum_r = 0.0f;
            int read_idx = write_idx;

            if (position_changed) {
                float alpha = static_cast<float>(i) / static_cast<float>(frames);
                float inv_alpha = 1.0f - alpha;
                
                for (int n = 0; n < hrir_len; ++n) {
                    float h_l = inv_alpha * h_l_old[n] + alpha * h_l_new[n];
                    float h_r = inv_alpha * h_r_old[n] + alpha * h_r_new[n];
                    
                    float hist_sample = history[read_idx];
                    sum_l += h_l * hist_sample;
                    sum_r += h_r * hist_sample;
                    
                    read_idx--;
                    if (read_idx < 0) read_idx += hrir_len;
                }
            } else {
                for (int n = 0; n < hrir_len; ++n) {
                    float hist_sample = history[read_idx];
                    sum_l += h_l_new[n] * hist_sample;
                    sum_r += h_r_new[n] * hist_sample;
                    
                    read_idx--;
                    if (read_idx < 0) read_idx += hrir_len;
                }
            }

            write_idx = (write_idx + 1) % hrir_len;

            output_left[i] = sum_l;
            output_right[i] = sum_r;
        }

        if (position_changed) {
            position_changed = false;
            std::copy(h_l_new.begin(), h_l_new.end(), h_l_old.begin());
            std::copy(h_r_new.begin(), h_r_new.end(), h_r_old.begin());
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
        .def("load_hrir_database", &BinauralProcessor::load_hrir_database)
        .def("set_position", &BinauralProcessor::set_position)
        .def("process", &BinauralProcessor::process_py);
}
