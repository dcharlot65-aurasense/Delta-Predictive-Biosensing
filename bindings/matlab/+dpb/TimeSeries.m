classdef TimeSeries < handle
%TIMESERIES Time series signal data
%   TS = DPB.TIMESERIES(DATA, SAMPLE_RATE) creates a time series from
%   DATA (samples x channels matrix) with the specified SAMPLE_RATE in Hz.

    properties (SetAccess = private)
        Handle      % Internal handle to C object
        SampleRate  % Sampling rate in Hz
    end

    properties (Dependent)
        Duration    % Duration in seconds
        NumSamples  % Number of samples
        NumChannels % Number of channels
    end

    methods
        function obj = TimeSeries(data, sampleRate)
            arguments
                data (:,:) single
                sampleRate (1,1) double {mustBePositive}
            end
            obj.Handle = dpb_mex('timeseries_new', data, sampleRate);
            obj.SampleRate = sampleRate;
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                dpb_mex('timeseries_free', obj.Handle);
            end
        end

        function d = get.Duration(obj)
            d = dpb_mex('timeseries_duration', obj.Handle);
        end

        function n = get.NumSamples(obj)
            n = dpb_mex('timeseries_num_samples', obj.Handle);
        end

        function n = get.NumChannels(obj)
            n = dpb_mex('timeseries_num_channels', obj.Handle);
        end

        function data = getData(obj)
            data = dpb_mex('timeseries_get_data', obj.Handle);
        end
    end
end
