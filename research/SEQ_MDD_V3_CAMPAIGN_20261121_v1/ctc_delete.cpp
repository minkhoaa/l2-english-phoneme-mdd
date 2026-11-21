// Exact CTC deletion likelihoods for a fixed token sequence, O(T * L).
#include <cmath>
#include <vector>

static inline double pluslog(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    if (b > a) { const double tmp = a; a = b; b = tmp; }
    return a + std::log1p(std::exp(b - a));
}

extern "C" void ctc_delete(const double* lp, int T, int V, const int* can, int L, double* out) {
    const int S = 2 * L + 1;
    std::vector<int> ext(S, 0);
    for (int i = 0; i < L; ++i) ext[2*i+1] = can[i];
    std::vector<double> alpha(static_cast<size_t>(T)*S, -INFINITY);
    alpha[0] = lp[0];
    alpha[1] = lp[ext[1]];
    for (int t = 1; t < T; ++t) {
        const auto* prev = &alpha[static_cast<size_t>(t-1)*S];
        auto* cur = &alpha[static_cast<size_t>(t)*S];
        const auto* frame = &lp[static_cast<size_t>(t)*V];
        for (int s = 0; s < S; ++s) {
            double a = prev[s];
            if (s > 0) a = pluslog(a, prev[s-1]);
            if (s > 1 && ext[s] && ext[s] != ext[s-2]) a = pluslog(a, prev[s-2]);
            cur[s] = a + frame[ext[s]];
        }
    }
    const auto* last = &alpha[static_cast<size_t>(T-1)*S];
    out[0] = pluslog(last[S-1], last[S-2]);
    for (int i = 0; i < L-1; ++i) out[i+1] = -INFINITY;
    out[L] = L==1 ? last[0] : pluslog(last[S-3], last[S-4]);

    // beta excludes emissions at t. Cross each deletion at the first frame of its right phone.
    // This enumerates each surviving path exactly once (even when the neighboring labels repeat).
    std::vector<double> beta(S, -INFINITY), prev(S, -INFINITY);
    beta[S-1] = beta[S-2] = 0.0;
    for (int t = T-1; t >= 0; --t) {
        const auto* frame = &lp[static_cast<size_t>(t)*V];
        if (t > 0) {
            const auto* a = &alpha[static_cast<size_t>(t-1)*S];
            for (int i = 0; i < L-1; ++i) {
                const int r = 2*i+3;
                double left = a[2*i];
                if (i > 0 && ext[r] != can[i-1]) left = pluslog(left, a[2*i-1]);
                out[i+1] = pluslog(out[i+1], left + frame[ext[r]] + beta[r]);
            }
        } else if (L > 1) {
            out[1] = pluslog(out[1], frame[ext[3]] + beta[3]);
        }
        for (int s = 0; s < S; ++s) {
            double b = beta[s] + frame[ext[s]];
            if (s+1 < S) b = pluslog(b, beta[s+1] + frame[ext[s+1]]);
            if (s+2 < S && ext[s+2] && ext[s+2] != ext[s])
                b = pluslog(b, beta[s+2] + frame[ext[s+2]]);
            prev[s] = b;
        }
        beta.swap(prev);
    }
}
