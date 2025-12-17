classdef LevelCrossingEncoder < handle
%LEVELCROSSINGENCODER Level crossing event encoder

    properties (SetAccess = private)
        Handle
        Threshold
    end

    methods
        function obj = LevelCrossingEncoder(options)
            arguments
                options.Threshold (1,1) double = 0.5
            end
            obj.Handle = dpb_mex('encoder_level_crossing_new', options.Threshold);
            obj.Threshold = options.Threshold;
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                dpb_mex('encoder_free', obj.Handle);
            end
        end

        function st = encode(obj, ts)
            arguments
                obj
                ts dpb.TimeSeries
            end
            handle = dpb_mex('encoder_encode', obj.Handle, ts.Handle);
            st = dpb.SpikeTrain(handle);
        end
    end
end
