classdef SpikeTrain < handle
%SPIKETRAIN Sequence of spike events

    properties (SetAccess = private)
        Handle
    end

    properties (Dependent)
        NumEvents
    end

    methods
        function obj = SpikeTrain(handle)
            if nargin > 0
                obj.Handle = handle;
            else
                obj.Handle = dpb_mex('spike_train_new');
            end
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                dpb_mex('spike_train_free', obj.Handle);
            end
        end

        function n = get.NumEvents(obj)
            n = dpb_mex('spike_train_len', obj.Handle);
        end

        function addEvent(obj, timestamp, channel, polarity)
            arguments
                obj
                timestamp (1,1) double
                channel (1,1) uint32
                polarity (1,1) int8 = int8(1)
            end
            dpb_mex('spike_train_add_event', obj.Handle, timestamp, channel, polarity);
        end

        function events = getEvents(obj)
            events = dpb_mex('spike_train_get_events', obj.Handle);
        end
    end
end
