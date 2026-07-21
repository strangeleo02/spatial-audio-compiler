#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <future>
#include <thread>
#include <atomic>
#include <memory>
#include <pybind11/stl.h>

#if defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define HAS_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

namespace py = pybind11;

typedef std::complex<float> Complex;

const double PI = 3.14159265358979323846;

// ─────────────────────────────────────────────────────────────────────────────
//  SIMD Vectorized Audio DSP Routines (AVX2: 8 float samples / cycle)
// ─────────────────────────────────────────────────────────────────────────────
inline void simd_scale_buffer(const float* src, float* dst, float scale, int size) {
    int i = 0;
#if defined(HAS_AVX2)
    __m256 v_scale = _mm256_set1_ps(scale);
    for (; i <= size - 8; i += 8) {
        __m256 v_src = _mm256_loadu_ps(src + i);
        __m256 v_res = _mm256_mul_ps(v_src, v_scale);
        _mm256_storeu_ps(dst + i, v_res);
    }
#elif defined(HAS_NEON)
    float32x4_t v_scale = vdupq_n_f32(scale);
    for (; i <= size - 4; i += 4) {
        float32x4_t v_src = vld1q_f32(src + i);
        float32x4_t v_res = vmulq_f32(v_src, v_scale);
        vst1q_f32(dst + i, v_res);
    }
#endif
    for (; i < size; ++i) {
        dst[i] = src[i] * scale;
    }
}

inline void simd_accumulate_channels(
    const std::vector<std::vector<float>>& temp_channels,
    float* out_channel,
    float master_vol,
    int num_tracks,
    int frames
) {
    int j = 0;
#if defined(HAS_AVX2)
    __m256 v_vol = _mm256_set1_ps(master_vol);
    for (; j <= frames - 8; j += 8) {
        __m256 v_acc = _mm256_setzero_ps();
        for (int i = 0; i < num_tracks; ++i) {
            __m256 v_track = _mm256_loadu_ps(&temp_channels[i][j]);
            v_acc = _mm256_add_ps(v_acc, v_track);
        }
        __m256 v_scaled = _mm256_mul_ps(v_acc, v_vol);
        _mm256_storeu_ps(out_channel + j, v_scaled);
    }
#elif defined(HAS_NEON)
    float32x4_t v_vol = vdupq_n_f32(master_vol);
    for (; j <= frames - 4; j += 4) {
        float32x4_t v_acc = vdupq_n_f32(0.0f);
        for (int i = 0; i < num_tracks; ++i) {
            float32x4_t v_track = vld1q_f32(&temp_channels[i][j]);
            v_acc = vaddq_f32(v_acc, v_track);
        }
        float32x4_t v_scaled = vmulq_f32(v_acc, v_vol);
        vst1q_f32(out_channel + j, v_scaled);
    }
#endif
    for (; j < frames; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < num_tracks; ++i) {
            sum += temp_channels[i][j];
        }
        out_channel[j] = sum * master_vol;
    }
}

inline uint32_t next_pow2(uint32_t n) {
    uint32_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Fast Cooley-Tukey Radix-2 FFT & IFFT Engine (O(N log N))
// ─────────────────────────────────────────────────────────────────────────────
inline void fft_radix2(std::vector<Complex>& x) {
    const size_t n = x.size();
    if (n <= 1) return;

    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * static_cast<float>(PI) / static_cast<float>(len);
        Complex wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                Complex u = x[i + j];
                Complex v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

inline void ifft_radix2(std::vector<Complex>& x) {
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) x[i] = std::conj(x[i]);
    fft_radix2(x);
    float scale = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::conj(x[i]) * scale;
    }
}

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
//  Binaural Processor using Partitioned FFT Overlap-Add HRTF Convolution
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

    // Cacheline-aligned Atomic Variables for Lock-Free UI / Audio Thread Separation
    alignas(64) std::atomic<float> target_azimuth{0.0f};
    alignas(64) std::atomic<float> target_elevation{0.0f};
    alignas(64) std::atomic<float> target_distance{1.0f};
    alignas(64) std::atomic<float> target_gain{1.0f};
    alignas(64) std::atomic<float> target_tone_lpf{20000.0f};
    alignas(64) std::atomic<bool> target_muted{false};
    alignas(64) std::atomic<bool> target_soloed{false};

    // Active DSP state (read/written exclusively by Audio Thread)
    double current_azimuth = 0.0;
    double current_elevation = 0.0;
    double current_distance = 1.0;
    double distance_gain = 1.0;
    double stem_gain = 1.0;
    double tone_cutoff = 20000.0;
    bool muted = false;
    bool soloed = false;

    // Active & target HRIR filter coefficients
    std::vector<float> h_l_new;
    std::vector<float> h_r_new;
    std::vector<float> h_l_old;
    std::vector<float> h_r_old;

    // Overlap-Add tail buffers for fast FFT block convolution
    std::vector<float> overlap_l;
    std::vector<float> overlap_r;

    bool position_changed = false;

    BiquadFilter tone_lpf;
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
        if (!overlap_l.empty()) std::fill(overlap_l.begin(), overlap_l.end(), 0.0f);
        if (!overlap_r.empty()) std::fill(overlap_r.begin(), overlap_r.end(), 0.0f);
        position_changed = false;
        target_azimuth.store(0.0f, std::memory_order_relaxed);
        target_elevation.store(0.0f, std::memory_order_relaxed);
        target_distance.store(1.0f, std::memory_order_relaxed);
        target_gain.store(1.0f, std::memory_order_relaxed);
        target_tone_lpf.store(20000.0f, std::memory_order_relaxed);
        target_muted.store(false, std::memory_order_relaxed);
        target_soloed.store(false, std::memory_order_relaxed);
        muted = false;
        soloed = false;
        if (hrir_len > 0) {
            update_position_internal(0.0, 0.0, 1.0);
            std::copy(h_l_new.begin(), h_l_new.end(), h_l_old.begin());
            std::copy(h_r_new.begin(), h_r_new.end(), h_r_old.begin());
        }
    }

    // Atomic Setters (Called by UI Thread — Non-Blocking, Lock-Free)
    void set_gain(double g) { target_gain.store(static_cast<float>(g), std::memory_order_relaxed); }
    void set_mute(bool m) { target_muted.store(m, std::memory_order_relaxed); }
    void set_solo(bool s) { target_soloed.store(s, std::memory_order_relaxed); }
    void set_tone_lpf(double cutoff_hz) { target_tone_lpf.store(static_cast<float>(cutoff_hz), std::memory_order_relaxed); }
    void set_position(double azimuth_deg, double elevation_deg, double distance_m) {
        target_azimuth.store(static_cast<float>(azimuth_deg), std::memory_order_relaxed);
        target_elevation.store(static_cast<float>(elevation_deg), std::memory_order_relaxed);
        target_distance.store(static_cast<float>(std::max(0.1, distance_m)), std::memory_order_relaxed);
    }

    void load_hrir_database(py::array_t<double> hrir_l, py::array_t<double> hrir_r) {
        py::buffer_info buf_l = hrir_l.request();
        py::buffer_info buf_r = hrir_r.request();

        if (buf_l.ndim != 3 || buf_r.ndim != 3) {
            throw std::runtime_error("HRIR database must be 3-dimensional array");
        }

        num_az = static_cast<int>(buf_l.shape[0]);
        num_el = static_cast<int>(buf_l.shape[1]);
        hrir_len = static_cast<int>(buf_l.shape[2]);

        db_hrir_l.assign(static_cast<const double*>(buf_l.ptr), static_cast<const double*>(buf_l.ptr) + buf_l.size);
        db_hrir_r.assign(static_cast<const double*>(buf_r.ptr), static_cast<const double*>(buf_r.ptr) + buf_r.size);

        h_l_new.resize(hrir_len, 0.0f);
        h_r_new.resize(hrir_len, 0.0f);
        h_l_old.resize(hrir_len, 0.0f);
        h_r_old.resize(hrir_len, 0.0f);

        overlap_l.assign(hrir_len, 0.0f);
        overlap_r.assign(hrir_len, 0.0f);

        update_position_internal(target_azimuth.load(std::memory_order_relaxed),
                                 target_elevation.load(std::memory_order_relaxed),
                                 target_distance.load(std::memory_order_relaxed));
        std::copy(h_l_new.begin(), h_l_new.end(), h_l_old.begin());
        std::copy(h_r_new.begin(), h_r_new.end(), h_r_old.begin());
    }

    void update_position_internal(double azimuth_deg, double elevation_deg, double distance_m) {
        double new_dist = std::max(0.1, distance_m);
        if (hrir_len > 0 &&
            std::abs(azimuth_deg - current_azimuth) < 1e-4 &&
            std::abs(elevation_deg - current_elevation) < 1e-4 &&
            std::abs(new_dist - current_distance) < 1e-4) {
            return;
        }

        current_azimuth = azimuth_deg;
        current_elevation = elevation_deg;
        current_distance = new_dist;

        distance_gain = 1.0 / current_distance;
        double lpf_freq = 20000.0 / (1.0 + 0.15 * (current_distance - 1.0));
        lpf_freq = std::clamp(lpf_freq, 1000.0, 20000.0);
        dist_lpf.set_lpf(lpf_freq, sample_rate);

        if (hrir_len <= 0) return;

        double az_rad = azimuth_deg * PI / 180.0;
        double el_rad = elevation_deg * PI / 180.0;

        double theta_ip_rad = std::asin(std::cos(el_rad) * std::sin(az_rad));
        double theta_ip_deg = theta_ip_rad * 180.0 / PI;

        double phi_ip_rad = std::atan2(std::sin(el_rad), std::cos(el_rad) * std::cos(az_rad));
        double phi_ip_deg = phi_ip_rad * 180.0 / PI;

        double az_val = std::clamp(theta_ip_deg, -80.0, 80.0);
        double el_val = std::clamp(phi_ip_deg, -45.0, 230.625);

        double el_idx_float = (el_val - (-45.0)) / 5.625;
        int el_idx_1 = std::clamp(static_cast<int>(std::floor(el_idx_float)), 0, num_el - 1);
        int el_idx_2 = std::clamp(el_idx_1 + 1, 0, num_el - 1);
        double el_frac = el_idx_float - el_idx_1;

        int az_idx_1 = 0;
        for (int i = 0; i < num_az - 1; ++i) {
            if (az_val >= cipic_az[i] && az_val <= cipic_az[i + 1]) {
                az_idx_1 = i;
                break;
            }
        }
        int az_idx_2 = std::min(az_idx_1 + 1, num_az - 1);
        double az_range = cipic_az[az_idx_2] - cipic_az[az_idx_1];
        double az_frac = (az_range > 1e-6) ? (az_val - cipic_az[az_idx_1]) / az_range : 0.0;

        double w11 = (1.0 - az_frac) * (1.0 - el_frac);
        double w21 = az_frac * (1.0 - el_frac);
        double w12 = (1.0 - az_frac) * el_frac;
        double w22 = az_frac * el_frac;

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
        // Lock-Free read of atomic target parameters pushed by UI thread
        float az = target_azimuth.load(std::memory_order_relaxed);
        float el = target_elevation.load(std::memory_order_relaxed);
        float dist = target_distance.load(std::memory_order_relaxed);
        float g = target_gain.load(std::memory_order_relaxed);
        float cutoff = target_tone_lpf.load(std::memory_order_relaxed);
        bool is_mute = target_muted.load(std::memory_order_relaxed);
        bool is_solo = target_soloed.load(std::memory_order_relaxed);

        stem_gain = g;
        muted = is_mute;
        soloed = is_solo;

        if (std::abs(cutoff - tone_cutoff) > 1.0) {
            tone_cutoff = cutoff;
            tone_lpf.set_lpf(tone_cutoff, sample_rate);
        }

        update_position_internal(az, el, dist);

        if (muted) {
            std::fill(output_left, output_left + frames, 0.0f);
            std::fill(output_right, output_right + frames, 0.0f);
            return;
        }

        if (hrir_len <= 0) {
            for (int i = 0; i < frames; ++i) {
                float sample = input[i] * static_cast<float>(stem_gain);
                float eq_sample = tone_lpf.process(sample);
                float dist_sample = dist_lpf.process(eq_sample) * static_cast<float>(distance_gain);
                output_left[i] = dist_sample;
                output_right[i] = dist_sample;
            }
            return;
        }

        // 1. Source -> Gain -> EQ LPF -> Distance Attenuation
        std::vector<float> mono_in(frames);
        for (int i = 0; i < frames; ++i) {
            float gain_sample = input[i] * static_cast<float>(stem_gain);
            float eq_sample = tone_lpf.process(gain_sample);
            mono_in[i] = dist_lpf.process(eq_sample) * static_cast<float>(distance_gain);
        }

        // 2. Overlap-Add FFT Convolution Engine
        // Input Block -> FFT -> Multiply HRIR FFT -> IFFT -> Overlap-Add -> Output
        uint32_t fft_size = next_pow2(frames + hrir_len - 1);

        // [STEP A] Input Block -> FFT
        std::vector<Complex> X(fft_size, Complex(0.0f, 0.0f));
        for (int i = 0; i < frames; ++i) {
            X[i] = Complex(mono_in[i], 0.0f);
        }
        fft_radix2(X);

        // Helper Lambda: FFT Convolution with HRIR filter taps
        auto compute_fft_conv = [&](const std::vector<float>& hrir_taps, std::vector<float>& out_conv) {
            // [STEP B] Forward FFT of HRIR filter taps
            std::vector<Complex> H(fft_size, Complex(0.0f, 0.0f));
            for (int n = 0; n < hrir_len; ++n) {
                H[n] = Complex(hrir_taps[n], 0.0f);
            }
            fft_radix2(H);

            // [STEP C] Multiply Input FFT by HRIR FFT: Y[k] = X[k] * H[k]
            std::vector<Complex> Y(fft_size);
            for (size_t k = 0; k < fft_size; ++k) {
                Y[k] = X[k] * H[k];
            }

            // [STEP D] IFFT: Inverse FFT back to time domain
            ifft_radix2(Y);

            out_conv.resize(fft_size);
            for (size_t k = 0; k < fft_size; ++k) {
                out_conv[k] = Y[k].real();
            }
        };

        std::vector<float> y_l_new, y_r_new;
        compute_fft_conv(h_l_new, y_l_new);
        compute_fft_conv(h_r_new, y_r_new);

        std::vector<float> y_l_old, y_r_old;
        if (position_changed) {
            compute_fft_conv(h_l_old, y_l_old);
            compute_fft_conv(h_r_old, y_r_old);
        }

        if (overlap_l.size() < static_cast<size_t>(hrir_len)) overlap_l.resize(hrir_len, 0.0f);
        if (overlap_r.size() < static_cast<size_t>(hrir_len)) overlap_r.resize(hrir_len, 0.0f);

        // [STEP E] Overlap Add synthesis
        for (int i = 0; i < frames; ++i) {
            float sample_l, sample_r;
            if (position_changed) {
                float alpha = static_cast<float>(i) / static_cast<float>(frames);
                float inv_alpha = 1.0f - alpha;
                sample_l = inv_alpha * y_l_old[i] + alpha * y_l_new[i];
                sample_r = inv_alpha * y_r_old[i] + alpha * y_r_new[i];
            } else {
                sample_l = y_l_new[i];
                sample_r = y_r_new[i];
            }

            // [STEP F] Output = Time-domain IFFT result + Overlap Tail from previous block
            output_left[i] = sample_l + overlap_l[i];
            output_right[i] = sample_r + overlap_r[i];
        }

        // Save Overlap Tail for Next Block
        for (int n = 0; n < hrir_len - 1; ++n) {
            float tail_l = (frames + n < static_cast<int>(y_l_new.size())) ? y_l_new[frames + n] : 0.0f;
            float tail_r = (frames + n < static_cast<int>(y_r_new.size())) ? y_r_new[frames + n] : 0.0f;
            if (position_changed) {
                float alpha = static_cast<float>(frames - 1) / static_cast<float>(frames);
                float inv_alpha = 1.0f - alpha;
                float old_l = (frames + n < static_cast<int>(y_l_old.size())) ? y_l_old[frames + n] : 0.0f;
                float old_r = (frames + n < static_cast<int>(y_r_old.size())) ? y_r_old[frames + n] : 0.0f;
                tail_l = inv_alpha * old_l + alpha * tail_l;
                tail_r = inv_alpha * old_r + alpha * tail_r;
            }
            overlap_l[n] = tail_l + (n + frames < static_cast<int>(overlap_l.size()) ? overlap_l[n + frames] : 0.0f);
            overlap_r[n] = tail_r + (n + frames < static_cast<int>(overlap_r.size()) ? overlap_r[n + frames] : 0.0f);
        }
        std::fill(overlap_l.begin() + (hrir_len - 1), overlap_l.end(), 0.0f);
        std::fill(overlap_r.begin() + (hrir_len - 1), overlap_r.end(), 0.0f);

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

// ─────────────────────────────────────────────────────────────────────────────
//  Lock-Free Single-Producer Single-Consumer (SPSC) Audio Ring Buffer
// ─────────────────────────────────────────────────────────────────────────────
class LockFreeAudioRingBuffer {
private:
    std::vector<float> buffer;
    size_t capacity;
    alignas(64) std::atomic<size_t> head{0}; // Write index (Producer / Mic)
    alignas(64) std::atomic<size_t> tail{0}; // Read index (Consumer / DSP)

public:
    LockFreeAudioRingBuffer(size_t cap = 65536) : capacity(cap) {
        // Pre-allocate buffer during construction (ZERO allocation during playback!)
        buffer.resize(capacity, 0.0f);
    }

    void reset() {
        head.store(0, std::memory_order_relaxed);
        tail.store(0, std::memory_order_relaxed);
        std::fill(buffer.begin(), buffer.end(), 0.0f);
    }

    size_t available_write() const {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_acquire);
        if (h >= t) return capacity - (h - t) - 1;
        return t - h - 1;
    }

    size_t available_read() const {
        size_t h = head.load(std::memory_order_acquire);
        size_t t = tail.load(std::memory_order_relaxed);
        if (h >= t) return h - t;
        return capacity - (t - h);
    }

    // Producer method (Mic / Live Input thread)
    bool push(py::array_t<float> input_samples) {
        py::buffer_info buf = input_samples.request();
        const float* ptr = static_cast<const float*>(buf.ptr);
        size_t count = static_cast<size_t>(buf.shape[0]);

        if (available_write() < count) return false; // Overrun protection

        size_t h = head.load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i) {
            buffer[(h + i) % capacity] = ptr[i];
        }
        head.store((h + count) % capacity, std::memory_order_release);
        return true;
    }

    // Consumer method (DSP thread)
    py::array_t<float> pop(size_t count) {
        auto out_arr = py::array_t<float>(count);
        float* out_ptr = static_cast<float*>(out_arr.request().ptr);

        size_t avail = available_read();
        size_t to_read = std::min(count, avail);

        size_t t = tail.load(std::memory_order_relaxed);
        for (size_t i = 0; i < to_read; ++i) {
            out_ptr[i] = buffer[(t + i) % capacity];
        }
        // Fill underrun tail with silence if underflow
        for (size_t i = to_read; i < count; ++i) {
            out_ptr[i] = 0.0f;
        }

        tail.store((t + to_read) % capacity, std::memory_order_release);
        return out_arr;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Multi-Threaded Parallel Binaural Audio Mixer (Scales across CPU cores)
// ─────────────────────────────────────────────────────────────────────────────
class ParallelBinauralMixer {
private:
    std::vector<std::shared_ptr<BinauralProcessor>> processors;
    double sample_rate;
    std::vector<std::vector<float>> temp_l;
    std::vector<std::vector<float>> temp_r;
    size_t prealloc_frames = 0;
    std::vector<const float*> input_ptrs_scratch;

public:
    ParallelBinauralMixer(int num_sources, double sr) : sample_rate(sr) {
        for (int i = 0; i < num_sources; ++i) {
            processors.push_back(std::make_shared<BinauralProcessor>(sr));
        }
        temp_l.resize(num_sources);
        temp_r.resize(num_sources);
        input_ptrs_scratch.resize(num_sources, nullptr);
    }

    std::shared_ptr<BinauralProcessor> get_processor(int index) {
        if (index < 0 || index >= static_cast<int>(processors.size())) {
            throw std::runtime_error("Invalid processor index");
        }
        return processors[index];
    }

    // Zero-allocation in-place processing into pre-allocated output buffers
    void process_mix_in_place(
        py::list input_arrays,
        py::array_t<float> output_left,
        py::array_t<float> output_right,
        double master_volume
    ) {
        int num_tracks = static_cast<int>(py::len(input_arrays));
        if (num_tracks != static_cast<int>(processors.size())) {
            throw std::runtime_error("Input array count does not match processor count");
        }

        py::buffer_info out_l_buf = output_left.request();
        py::buffer_info out_r_buf = output_right.request();
        int frames = static_cast<int>(out_l_buf.shape[0]);
        float* out_L = static_cast<float*>(out_l_buf.ptr);
        float* out_R = static_cast<float*>(out_r_buf.ptr);

        if (input_ptrs_scratch.size() < static_cast<size_t>(num_tracks)) {
            input_ptrs_scratch.resize(num_tracks, nullptr);
        }

        for (int i = 0; i < num_tracks; ++i) {
            py::array_t<float> arr = py::cast<py::array_t<float>>(input_arrays[i]);
            input_ptrs_scratch[i] = static_cast<const float*>(arr.request().ptr);
        }

        if (static_cast<size_t>(frames) > prealloc_frames) {
            prealloc_frames = static_cast<size_t>(frames);
            for (int i = 0; i < num_tracks; ++i) {
                temp_l[i].resize(prealloc_frames, 0.0f);
                temp_r[i].resize(prealloc_frames, 0.0f);
            }
        }

        {
            py::gil_scoped_release release;

            std::vector<std::future<void>> futures;
            futures.reserve(num_tracks);
            for (int i = 0; i < num_tracks; ++i) {
                const float* in_ptr = input_ptrs_scratch[i];
                float* out_l = temp_l[i].data();
                float* out_r = temp_r[i].data();
                auto proc = processors[i];

                futures.push_back(std::async(std::launch::async, [proc, in_ptr, out_l, out_r, frames]() {
                    proc->process(in_ptr, out_l, out_r, frames);
                }));
            }

            for (auto& f : futures) {
                f.get();
            }
        }

        float m_vol = static_cast<float>(master_volume);
        simd_accumulate_channels(temp_l, out_L, m_vol, num_tracks, frames);
        simd_accumulate_channels(temp_r, out_R, m_vol, num_tracks, frames);
    }

    std::pair<py::array_t<float>, py::array_t<float>> process_mix_parallel(
        py::list input_arrays, double master_volume
    ) {
        int num_tracks = static_cast<int>(py::len(input_arrays));
        if (num_tracks != static_cast<int>(processors.size())) {
            throw std::runtime_error("Input array count does not match processor count");
        }

        int frames = 0;
        std::vector<const float*> input_ptrs(num_tracks, nullptr);
        for (int i = 0; i < num_tracks; ++i) {
            py::array_t<float> arr = py::cast<py::array_t<float>>(input_arrays[i]);
            py::buffer_info info = arr.request();
            if (i == 0) frames = static_cast<int>(info.shape[0]);
            input_ptrs[i] = static_cast<const float*>(info.ptr);
        }

        if (static_cast<size_t>(frames) > prealloc_frames) {
            prealloc_frames = static_cast<size_t>(frames);
            for (int i = 0; i < num_tracks; ++i) {
                temp_l[i].resize(prealloc_frames, 0.0f);
                temp_r[i].resize(prealloc_frames, 0.0f);
            }
        }

        {
            py::gil_scoped_release release;

            std::vector<std::future<void>> futures;
            for (int i = 0; i < num_tracks; ++i) {
                const float* in_ptr = input_ptrs[i];
                float* out_l = temp_l[i].data();
                float* out_r = temp_r[i].data();
                auto proc = processors[i];

                futures.push_back(std::async(std::launch::async, [proc, in_ptr, out_l, out_r, frames]() {
                    proc->process(in_ptr, out_l, out_r, frames);
                }));
            }

            for (auto& f : futures) {
                f.get();
            }
        }

        auto output_left = py::array_t<float>(frames);
        auto output_right = py::array_t<float>(frames);
        float* out_L = static_cast<float*>(output_left.request().ptr);
        float* out_R = static_cast<float*>(output_right.request().ptr);

        float m_vol = static_cast<float>(master_volume);
        simd_accumulate_channels(temp_l, out_L, m_vol, num_tracks, frames);
        simd_accumulate_channels(temp_r, out_R, m_vol, num_tracks, frames);

        return {output_left, output_right};
    }
};

PYBIND11_MODULE(binaural_dsp, m) {
    m.doc() = "C++ High Performance Multi-Threaded Binaural 3D Spatial DSP Engine";

    py::class_<BinauralProcessor, std::shared_ptr<BinauralProcessor>>(m, "BinauralProcessor")
        .def(py::init<double>())
        .def("reset", &BinauralProcessor::reset)
        .def("load_hrir_database", &BinauralProcessor::load_hrir_database)
        .def("set_position", &BinauralProcessor::set_position)
        .def("set_gain", &BinauralProcessor::set_gain)
        .def("set_mute", &BinauralProcessor::set_mute)
        .def("set_solo", &BinauralProcessor::set_solo)
        .def("set_tone_lpf", &BinauralProcessor::set_tone_lpf)
        .def("process", &BinauralProcessor::process_py);

    py::class_<ParallelBinauralMixer>(m, "ParallelBinauralMixer")
        .def(py::init<int, double>())
        .def("get_processor", &ParallelBinauralMixer::get_processor)
        .def("process_mix_parallel", &ParallelBinauralMixer::process_mix_parallel)
        .def("process_mix_in_place", &ParallelBinauralMixer::process_mix_in_place);

    py::class_<LockFreeAudioRingBuffer>(m, "LockFreeAudioRingBuffer")
        .def(py::init<size_t>(), py::arg("capacity") = 65536)
        .def("reset", &LockFreeAudioRingBuffer::reset)
        .def("available_write", &LockFreeAudioRingBuffer::available_write)
        .def("available_read", &LockFreeAudioRingBuffer::available_read)
        .def("push", &LockFreeAudioRingBuffer::push)
        .def("pop", &LockFreeAudioRingBuffer::pop);
}
