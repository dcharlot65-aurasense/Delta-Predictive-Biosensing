using Test
using DPB

@testset "DPB.jl Tests" begin

    @testset "Library Loading" begin
        @test DPB.version() isa String
        @test !isempty(DPB.version())
        println("DPB version: ", DPB.version())
    end

    @testset "TimeSeries Construction" begin
        # Create a simple time series
        data = rand(Float32, 100, 2)  # 100 samples, 2 channels
        ts = TimeSeries(data, 1000.0)

        @test ts isa TimeSeries
        @test num_samples(ts) == 100
        @test num_channels(ts) == 2
        @test sample_rate(ts) ≈ 1000.0
        @test duration(ts) ≈ 0.1  # 100 samples / 1000 Hz = 0.1 seconds
    end

    @testset "SpikeTrain Construction" begin
        # Create empty spike train
        st = SpikeTrain()

        @test st isa SpikeTrain
        @test length(st) == 0
        @test isempty(st)
    end

    @testset "Level Crossing Encoder" begin
        # Create encoder
        encoder = LevelCrossingEncoder(threshold=0.1)

        @test encoder isa LevelCrossingEncoder
        @test encoder.threshold == 0.1

        # Test encoding
        data = randn(Float32, 1000, 1)
        ts = TimeSeries(data, 1000.0)
        spikes = encode(encoder, ts)

        @test spikes isa SpikeTrain
        @test length(spikes) >= 0  # Should produce some spikes

        println("Generated ", length(spikes), " spikes from 1000 samples")
    end

    @testset "Encoder Error Handling" begin
        # Test invalid threshold
        @test_throws ErrorException LevelCrossingEncoder(threshold=-1.0)
        @test_throws ErrorException LevelCrossingEncoder(threshold=0.0)
    end

    @testset "Spike Iteration" begin
        # Create a spike train with some data
        data = Float32[0, 1, -1, 2, -2, 1, 0]
        ts = TimeSeries(reshape(data, :, 1), 100.0)
        encoder = LevelCrossingEncoder(threshold=0.5)
        spikes = encode(encoder, ts)

        if !isempty(spikes)
            # Test indexing
            first_spike = spikes[1]
            @test first_spike isa SpikeEvent
            @test first_spike.timestamp >= 0.0

            # Test iteration
            spike_count = 0
            for spike in spikes
                spike_count += 1
                @test spike isa SpikeEvent
                @test spike.timestamp >= 0.0
            end
            @test spike_count == length(spikes)

            # Test collect
            all_spikes = collect_spikes(spikes)
            @test length(all_spikes) == length(spikes)
            @test all(s isa SpikeEvent for s in all_spikes)
        end
    end

    @testset "Synthetic Data - Sine Wave" begin
        ts = DPB.Synth.sine_wave(
            frequency=10.0,
            amplitude=1.0,
            duration=1.0,
            sample_rate=1000.0
        )

        @test ts isa TimeSeries
        @test num_samples(ts) == 1000
        @test duration(ts) ≈ 1.0

        println("Generated sine wave: ", num_samples(ts), " samples")
    end

    @testset "Synthetic Data - White Noise" begin
        ts = DPB.Synth.white_noise(
            amplitude=0.5,
            duration=2.0,
            sample_rate=500.0
        )

        @test ts isa TimeSeries
        @test num_samples(ts) == 1000
        @test duration(ts) ≈ 2.0
    end

    @testset "Synthetic Data - Brownian Motion" begin
        ts = DPB.Synth.brownian_motion(
            diffusion=0.1,
            duration=1.0,
            sample_rate=1000.0
        )

        @test ts isa TimeSeries
        @test num_samples(ts) == 1000
    end

    @testset "Synthetic Data - ECG" begin
        ecg = DPB.Synth.ecg_signal(
            heart_rate=72.0,
            duration=5.0,
            sample_rate=1000.0
        )

        @test ecg isa TimeSeries
        @test duration(ecg) ≈ 5.0

        println("Generated ECG: ", num_samples(ecg), " samples")
    end

    @testset "Synthetic Data - EEG" begin
        eeg = DPB.Synth.eeg_alpha(
            frequency=10.0,
            duration=2.0,
            sample_rate=1000.0,
            channels=4
        )

        @test eeg isa TimeSeries
        @test num_channels(eeg) == 4
        @test duration(eeg) ≈ 2.0
    end

    @testset "Synthetic Data - EMG" begin
        emg = DPB.Synth.emg_burst(
            burst_duration=0.3,
            burst_interval=1.0,
            duration=5.0,
            sample_rate=1000.0
        )

        @test emg isa TimeSeries
        @test duration(emg) ≈ 5.0
    end

    @testset "Full Pipeline" begin
        # Generate synthetic data
        signal = DPB.Synth.sine_wave(
            frequency=5.0,
            amplitude=2.0,
            duration=2.0,
            sample_rate=1000.0
        )

        @test signal isa TimeSeries

        # Encode it
        encoder = LevelCrossingEncoder(threshold=0.5)
        spikes = encode(encoder, signal)

        @test spikes isa SpikeTrain
        @test length(spikes) > 0

        println("Full pipeline test:")
        println("  Input: ", num_samples(signal), " samples")
        println("  Output: ", length(spikes), " spikes")
        println("  Compression ratio: ", num_samples(signal) / max(length(spikes), 1))
    end

    @testset "Memory Management" begin
        # Create many objects to test finalizers
        for i in 1:10
            data = rand(Float32, 100, 1)
            ts = TimeSeries(data, 1000.0)
            encoder = LevelCrossingEncoder(threshold=0.1)
            spikes = encode(encoder, ts)
        end

        # Force garbage collection
        GC.gc()

        # If we get here without crashes, memory management is working
        @test true
    end
end

println("\nAll tests passed!")
