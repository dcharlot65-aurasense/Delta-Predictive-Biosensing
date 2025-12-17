% DPB - Delta-Predictive Biosensing Framework
%
% Classes:
%   TimeSeries          - Time series signal data
%   SpikeTrain          - Sequence of spike events
%   LevelCrossingEncoder - Level crossing event encoder
%
% Functions:
%   dpb.version         - Get library version
%
% Example:
%   signal = dpb.TimeSeries(data, 250.0);  % 250 Hz sample rate
%   encoder = dpb.LevelCrossingEncoder('Threshold', 0.5);
%   events = encoder.encode(signal);
