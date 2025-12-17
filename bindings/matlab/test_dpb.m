function test_dpb()
%TEST_DPB Test the DPB MATLAB bindings
%   This script tests the basic functionality of the DPB framework
%   MATLAB bindings.

    fprintf('Testing DPB MATLAB Bindings\n');
    fprintf('===========================\n\n');

    % Test 1: Version
    fprintf('Test 1: Library Version\n');
    try
        v = dpb.version();
        fprintf('  Version: %s\n', v);
        fprintf('  PASSED\n\n');
    catch ME
        fprintf('  FAILED: %s\n\n', ME.message);
        return;
    end

    % Test 2: TimeSeries creation
    fprintf('Test 2: TimeSeries Creation\n');
    try
        % Create synthetic ECG-like signal
        sample_rate = 250.0;  % Hz
        duration = 2.0;  % seconds
        num_samples = round(sample_rate * duration);

        % Generate simple sine wave
        t = linspace(0, duration, num_samples)';
        data = single(sin(2*pi*1.5*t));  % 1.5 Hz sine wave

        ts = dpb.TimeSeries(data, sample_rate);
        fprintf('  Created TimeSeries:\n');
        fprintf('    Sample Rate: %.1f Hz\n', ts.SampleRate);
        fprintf('    Duration: %.2f s\n', ts.Duration);
        fprintf('    Num Samples: %d\n', ts.NumSamples);
        fprintf('    Num Channels: %d\n', ts.NumChannels);
        fprintf('  PASSED\n\n');
    catch ME
        fprintf('  FAILED: %s\n\n', ME.message);
        return;
    end

    % Test 3: Multi-channel TimeSeries
    fprintf('Test 3: Multi-channel TimeSeries\n');
    try
        % Create 3-channel signal
        data_multi = single([sin(2*pi*1.0*t), ...
                             sin(2*pi*2.0*t), ...
                             sin(2*pi*3.0*t)]);

        ts_multi = dpb.TimeSeries(data_multi, sample_rate);
        fprintf('  Created Multi-channel TimeSeries:\n');
        fprintf('    Num Channels: %d\n', ts_multi.NumChannels);
        fprintf('  PASSED\n\n');
    catch ME
        fprintf('  FAILED: %s\n\n', ME.message);
        return;
    end

    % Test 4: SpikeTrain creation
    fprintf('Test 4: SpikeTrain Creation\n');
    try
        st = dpb.SpikeTrain();
        fprintf('  Created SpikeTrain\n');
        fprintf('    Initial Num Events: %d\n', st.NumEvents);

        % Add some events
        st.addEvent(0.1, uint32(0), int8(1));
        st.addEvent(0.2, uint32(0), int8(-1));
        st.addEvent(0.3, uint32(1), int8(1));

        fprintf('    After adding 3 events: %d\n', st.NumEvents);
        fprintf('  PASSED\n\n');
    catch ME
        fprintf('  FAILED: %s\n\n', ME.message);
        return;
    end

    % Test 5: Level Crossing Encoder
    fprintf('Test 5: Level Crossing Encoder\n');
    try
        encoder = dpb.LevelCrossingEncoder('Threshold', 0.1);
        fprintf('  Created encoder with threshold: %.2f\n', encoder.Threshold);

        % Encode the signal
        events = encoder.encode(ts);
        fprintf('  Encoded signal to %d events\n', events.NumEvents);
        fprintf('  PASSED\n\n');
    catch ME
        fprintf('  FAILED: %s\n\n', ME.message);
        return;
    end

    % Test 6: Different threshold values
    fprintf('Test 6: Encoding with Different Thresholds\n');
    try
        thresholds = [0.05, 0.1, 0.2, 0.5];
        for i = 1:length(thresholds)
            enc = dpb.LevelCrossingEncoder('Threshold', thresholds(i));
            evt = enc.encode(ts);
            fprintf('  Threshold %.2f: %d events\n', thresholds(i), evt.NumEvents);
        end
        fprintf('  PASSED\n\n');
    catch ME
        fprintf('  FAILED: %s\n\n', ME.message);
        return;
    end

    % Test 7: Memory cleanup
    fprintf('Test 7: Memory Cleanup\n');
    try
        % Create and destroy multiple objects
        for i = 1:10
            temp_data = single(randn(100, 1));
            temp_ts = dpb.TimeSeries(temp_data, 100.0);
            temp_enc = dpb.LevelCrossingEncoder('Threshold', 0.1);
            temp_evt = temp_enc.encode(temp_ts);
            % Objects should be automatically cleaned up
        end
        fprintf('  Created and destroyed 10 sets of objects\n');
        fprintf('  PASSED\n\n');
    catch ME
        fprintf('  FAILED: %s\n\n', ME.message);
        return;
    end

    fprintf('===========================\n');
    fprintf('All tests completed successfully!\n');
end
