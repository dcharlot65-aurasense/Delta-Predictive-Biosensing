#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <stdint.h>

// Function declarations from dpb-ffi
extern const char* dpb_version(void);
extern void* dpb_timeseries_new(const float* data, size_t samples, size_t channels, double sample_rate);
extern void dpb_timeseries_free(void* ts);
extern double dpb_timeseries_duration(const void* ts);
extern size_t dpb_timeseries_num_samples(const void* ts);
extern size_t dpb_timeseries_num_channels(const void* ts);
extern const float* dpb_timeseries_get_data(const void* ts);
extern void* dpb_spike_train_new(void);
extern void dpb_spike_train_free(void* st);
extern void dpb_spike_train_add_event(void* st, double timestamp, uint32_t channel, int8_t polarity);
extern size_t dpb_spike_train_len(const void* st);
extern void* dpb_encoder_level_crossing_new(double threshold);
extern void dpb_encoder_free(void* enc);
extern void* dpb_encoder_encode(const void* enc, const void* ts);
extern const char* dpb_last_error(void);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("DPB:InvalidInput", "First argument must be command string");
    }

    char cmd[64];
    mxGetString(prhs[0], cmd, sizeof(cmd));

    if (strcmp(cmd, "version") == 0) {
        const char* v = dpb_version();
        plhs[0] = mxCreateString(v);
    }
    else if (strcmp(cmd, "timeseries_new") == 0) {
        // Get data matrix and sample rate
        float* data = (float*)mxGetData(prhs[1]);
        size_t samples = mxGetM(prhs[1]);
        size_t channels = mxGetN(prhs[1]);
        double sample_rate = mxGetScalar(prhs[2]);

        void* ts = dpb_timeseries_new(data, samples, channels, sample_rate);
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)ts;
    }
    else if (strcmp(cmd, "timeseries_free") == 0) {
        uint64_t handle = *((uint64_t*)mxGetData(prhs[1]));
        dpb_timeseries_free((void*)handle);
    }
    else if (strcmp(cmd, "timeseries_duration") == 0) {
        uint64_t handle = *((uint64_t*)mxGetData(prhs[1]));
        plhs[0] = mxCreateDoubleScalar(dpb_timeseries_duration((void*)handle));
    }
    else if (strcmp(cmd, "timeseries_num_samples") == 0) {
        uint64_t handle = *((uint64_t*)mxGetData(prhs[1]));
        plhs[0] = mxCreateDoubleScalar((double)dpb_timeseries_num_samples((void*)handle));
    }
    else if (strcmp(cmd, "timeseries_num_channels") == 0) {
        uint64_t handle = *((uint64_t*)mxGetData(prhs[1]));
        plhs[0] = mxCreateDoubleScalar((double)dpb_timeseries_num_channels((void*)handle));
    }
    else if (strcmp(cmd, "spike_train_new") == 0) {
        void* st = dpb_spike_train_new();
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)st;
    }
    else if (strcmp(cmd, "spike_train_free") == 0) {
        uint64_t handle = *((uint64_t*)mxGetData(prhs[1]));
        dpb_spike_train_free((void*)handle);
    }
    else if (strcmp(cmd, "spike_train_len") == 0) {
        uint64_t handle = *((uint64_t*)mxGetData(prhs[1]));
        plhs[0] = mxCreateDoubleScalar((double)dpb_spike_train_len((void*)handle));
    }
    else if (strcmp(cmd, "encoder_level_crossing_new") == 0) {
        double threshold = mxGetScalar(prhs[1]);
        void* enc = dpb_encoder_level_crossing_new(threshold);
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)enc;
    }
    else if (strcmp(cmd, "encoder_free") == 0) {
        uint64_t handle = *((uint64_t*)mxGetData(prhs[1]));
        dpb_encoder_free((void*)handle);
    }
    else if (strcmp(cmd, "encoder_encode") == 0) {
        uint64_t enc_handle = *((uint64_t*)mxGetData(prhs[1]));
        uint64_t ts_handle = *((uint64_t*)mxGetData(prhs[2]));
        void* st = dpb_encoder_encode((void*)enc_handle, (void*)ts_handle);
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)st;
    }
    else {
        mexErrMsgIdAndTxt("DPB:UnknownCommand", "Unknown command: %s", cmd);
    }
}
