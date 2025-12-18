/**
 * @file dpb_r.c
 * @brief R <-> DPB FFI Bridge
 *
 * C bridge code for R integration with the DPB Framework.
 * Uses R's .Call interface for high-performance native calls.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>
#include <string.h>

/* Include the DPB FFI header */
#include "dpb.h"

/* ============================================================================
 * Version and Error Handling
 * ============================================================================ */

SEXP C_dpb_version(void) {
    const char* ver = dpb_version();
    return ver ? Rf_mkString(ver) : R_NilValue;
}

SEXP C_dpb_last_error(void) {
    const char* err = dpb_last_error();
    return err ? Rf_mkString(err) : R_NilValue;
}

SEXP C_dpb_clear_error(void) {
    dpb_clear_error();
    return R_NilValue;
}

/* ============================================================================
 * TimeSeries Functions
 * ============================================================================ */

SEXP C_dpb_timeseries_new(SEXP data, SEXP num_samples, SEXP num_channels, SEXP sample_rate) {
    size_t ns = (size_t)Rf_asInteger(num_samples);
    size_t nc = (size_t)Rf_asInteger(num_channels);
    double sr = Rf_asReal(sample_rate);
    size_t total = ns * nc;

    /* R stores doubles, we need floats */
    float* float_data = (float*)R_alloc(total, sizeof(float));
    double* r_data = REAL(data);

    for (size_t i = 0; i < total; i++) {
        float_data[i] = (float)r_data[i];
    }

    DpbTimeSeries* ts = dpb_timeseries_new(float_data, ns, nc, sr);

    if (!ts) {
        return R_NilValue;
    }

    /* Create external pointer with destructor */
    SEXP ptr = PROTECT(R_MakeExternalPtr(ts, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP C_dpb_timeseries_free(SEXP ptr) {
    if (TYPEOF(ptr) == EXTPTRSXP) {
        DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
        if (ts) {
            dpb_timeseries_free(ts);
            R_ClearExternalPtr(ptr);
        }
    }
    return R_NilValue;
}

SEXP C_dpb_timeseries_duration(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return Rf_ScalarReal(NA_REAL);
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    if (!ts) return Rf_ScalarReal(NA_REAL);
    return Rf_ScalarReal(dpb_timeseries_duration(ts));
}

SEXP C_dpb_timeseries_num_samples(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return Rf_ScalarInteger(NA_INTEGER);
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    if (!ts) return Rf_ScalarInteger(NA_INTEGER);
    return Rf_ScalarInteger((int)dpb_timeseries_num_samples(ts));
}

SEXP C_dpb_timeseries_num_channels(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return Rf_ScalarInteger(NA_INTEGER);
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    if (!ts) return Rf_ScalarInteger(NA_INTEGER);
    return Rf_ScalarInteger((int)dpb_timeseries_num_channels(ts));
}

SEXP C_dpb_timeseries_sample_rate(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return Rf_ScalarReal(NA_REAL);
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    if (!ts) return Rf_ScalarReal(NA_REAL);
    return Rf_ScalarReal(dpb_timeseries_sample_rate(ts));
}

SEXP C_dpb_timeseries_get_data(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return R_NilValue;
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    if (!ts) return R_NilValue;

    size_t ns = dpb_timeseries_num_samples(ts);
    size_t nc = dpb_timeseries_num_channels(ts);
    const float* data = dpb_timeseries_get_data(ts);

    if (!data) return R_NilValue;

    size_t total = ns * nc;
    SEXP result = PROTECT(Rf_allocVector(REALSXP, total));
    double* rdata = REAL(result);

    /* Convert float to double */
    for (size_t i = 0; i < total; i++) {
        rdata[i] = (double)data[i];
    }

    UNPROTECT(1);
    return result;
}

/* ============================================================================
 * SpikeTrain Functions
 * ============================================================================ */

SEXP C_dpb_spike_train_new(SEXP num_channels) {
    uint32_t nc = (uint32_t)Rf_asInteger(num_channels);
    DpbSpikeTrain* st = dpb_spike_train_new(nc);

    if (!st) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(st, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP C_dpb_spike_train_free(SEXP ptr) {
    if (TYPEOF(ptr) == EXTPTRSXP) {
        DpbSpikeTrain* st = (DpbSpikeTrain*)R_ExternalPtrAddr(ptr);
        if (st) {
            dpb_spike_train_free(st);
            R_ClearExternalPtr(ptr);
        }
    }
    return R_NilValue;
}

SEXP C_dpb_spike_train_add_event(SEXP ptr, SEXP timestamp, SEXP channel, SEXP polarity, SEXP magnitude) {
    if (TYPEOF(ptr) != EXTPTRSXP) return Rf_ScalarInteger(DPB_ERROR_NULL_POINTER);
    DpbSpikeTrain* st = (DpbSpikeTrain*)R_ExternalPtrAddr(ptr);
    if (!st) return Rf_ScalarInteger(DPB_ERROR_NULL_POINTER);

    double ts = Rf_asReal(timestamp);
    uint32_t ch = (uint32_t)Rf_asInteger(channel);
    int8_t pol = (int8_t)Rf_asInteger(polarity);
    /* magnitude is stored in polarity for simplicity in current API */

    int result = dpb_spike_train_add_event(st, ts, ch, pol);
    return Rf_ScalarInteger(result);
}

SEXP C_dpb_spike_train_len(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return Rf_ScalarInteger(0);
    DpbSpikeTrain* st = (DpbSpikeTrain*)R_ExternalPtrAddr(ptr);
    if (!st) return Rf_ScalarInteger(0);
    return Rf_ScalarInteger((int)dpb_spike_train_len(st));
}

SEXP C_dpb_spike_train_num_channels(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) return Rf_ScalarInteger(0);
    DpbSpikeTrain* st = (DpbSpikeTrain*)R_ExternalPtrAddr(ptr);
    if (!st) return Rf_ScalarInteger(0);
    return Rf_ScalarInteger((int)dpb_spike_train_num_channels(st));
}

SEXP C_dpb_spike_train_get_event(SEXP ptr, SEXP index) {
    if (TYPEOF(ptr) != EXTPTRSXP) return R_NilValue;
    DpbSpikeTrain* st = (DpbSpikeTrain*)R_ExternalPtrAddr(ptr);
    if (!st) return R_NilValue;

    size_t idx = (size_t)Rf_asInteger(index);

    double timestamp;
    uint32_t channel;
    int8_t polarity;
    float magnitude;

    int result = dpb_spike_train_get_event(st, idx, &timestamp, &channel, &polarity, &magnitude);

    if (result != DPB_SUCCESS) {
        return R_NilValue;
    }

    /* Return as a named list */
    SEXP event = PROTECT(Rf_allocVector(VECSXP, 4));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));

    SET_VECTOR_ELT(event, 0, Rf_ScalarReal(timestamp));
    SET_VECTOR_ELT(event, 1, Rf_ScalarInteger((int)channel));
    SET_VECTOR_ELT(event, 2, Rf_ScalarInteger((int)polarity));
    SET_VECTOR_ELT(event, 3, Rf_ScalarReal((double)magnitude));

    SET_STRING_ELT(names, 0, Rf_mkChar("timestamp"));
    SET_STRING_ELT(names, 1, Rf_mkChar("channel"));
    SET_STRING_ELT(names, 2, Rf_mkChar("polarity"));
    SET_STRING_ELT(names, 3, Rf_mkChar("magnitude"));

    Rf_setAttrib(event, R_NamesSymbol, names);

    UNPROTECT(2);
    return event;
}

/* ============================================================================
 * Encoder Functions
 * ============================================================================ */

SEXP C_dpb_encoder_level_crossing_new(SEXP threshold) {
    double th = Rf_asReal(threshold);
    DpbEncoder* enc = dpb_encoder_level_crossing_new(th);

    if (!enc) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(enc, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

/* Delta encoder - uses same underlying type with different params */
SEXP C_dpb_encoder_delta_new(SEXP threshold) {
    /* For now, use level crossing as delta encoder is not yet in FFI */
    /* This is a placeholder - real implementation would call dpb_encoder_delta_new */
    double th = Rf_asReal(threshold);
    DpbEncoder* enc = dpb_encoder_level_crossing_new(th);

    if (!enc) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(enc, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP C_dpb_encoder_free(SEXP ptr) {
    if (TYPEOF(ptr) == EXTPTRSXP) {
        DpbEncoder* enc = (DpbEncoder*)R_ExternalPtrAddr(ptr);
        if (enc) {
            dpb_encoder_free(enc);
            R_ClearExternalPtr(ptr);
        }
    }
    return R_NilValue;
}

SEXP C_dpb_encoder_encode(SEXP enc_ptr, SEXP ts_ptr) {
    if (TYPEOF(enc_ptr) != EXTPTRSXP || TYPEOF(ts_ptr) != EXTPTRSXP) {
        return R_NilValue;
    }

    DpbEncoder* enc = (DpbEncoder*)R_ExternalPtrAddr(enc_ptr);
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ts_ptr);

    if (!enc || !ts) {
        return R_NilValue;
    }

    DpbSpikeTrain* st = dpb_encoder_encode(enc, ts);

    if (!st) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(st, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

/* ============================================================================
 * Registration
 * ============================================================================ */

static const R_CallMethodDef CallEntries[] = {
    /* Version and errors */
    {"C_dpb_version", (DL_FUNC) &C_dpb_version, 0},
    {"C_dpb_last_error", (DL_FUNC) &C_dpb_last_error, 0},
    {"C_dpb_clear_error", (DL_FUNC) &C_dpb_clear_error, 0},

    /* TimeSeries */
    {"C_dpb_timeseries_new", (DL_FUNC) &C_dpb_timeseries_new, 4},
    {"C_dpb_timeseries_free", (DL_FUNC) &C_dpb_timeseries_free, 1},
    {"C_dpb_timeseries_duration", (DL_FUNC) &C_dpb_timeseries_duration, 1},
    {"C_dpb_timeseries_num_samples", (DL_FUNC) &C_dpb_timeseries_num_samples, 1},
    {"C_dpb_timeseries_num_channels", (DL_FUNC) &C_dpb_timeseries_num_channels, 1},
    {"C_dpb_timeseries_sample_rate", (DL_FUNC) &C_dpb_timeseries_sample_rate, 1},
    {"C_dpb_timeseries_get_data", (DL_FUNC) &C_dpb_timeseries_get_data, 1},

    /* SpikeTrain */
    {"C_dpb_spike_train_new", (DL_FUNC) &C_dpb_spike_train_new, 1},
    {"C_dpb_spike_train_free", (DL_FUNC) &C_dpb_spike_train_free, 1},
    {"C_dpb_spike_train_add_event", (DL_FUNC) &C_dpb_spike_train_add_event, 5},
    {"C_dpb_spike_train_len", (DL_FUNC) &C_dpb_spike_train_len, 1},
    {"C_dpb_spike_train_num_channels", (DL_FUNC) &C_dpb_spike_train_num_channels, 1},
    {"C_dpb_spike_train_get_event", (DL_FUNC) &C_dpb_spike_train_get_event, 2},

    /* Encoders */
    {"C_dpb_encoder_level_crossing_new", (DL_FUNC) &C_dpb_encoder_level_crossing_new, 1},
    {"C_dpb_encoder_delta_new", (DL_FUNC) &C_dpb_encoder_delta_new, 1},
    {"C_dpb_encoder_free", (DL_FUNC) &C_dpb_encoder_free, 1},
    {"C_dpb_encoder_encode", (DL_FUNC) &C_dpb_encoder_encode, 2},

    {NULL, NULL, 0}
};

void R_init_dpb(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
