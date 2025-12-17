function build_mex()
%BUILD_MEX Build the DPB MEX interface

    % Find the dpb-ffi library
    dpb_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    lib_path = fullfile(dpb_root, 'target', 'release');

    % Compiler flags
    if ispc
        lib_file = fullfile(lib_path, 'dpb_ffi.dll');
    elseif ismac
        lib_file = fullfile(lib_path, 'libdpb_ffi.dylib');
    else
        lib_file = fullfile(lib_path, 'libdpb_ffi.so');
    end

    if ~exist(lib_file, 'file')
        error('DPB library not found. Build with: cargo build --release -p dpb-ffi');
    end

    % Build MEX
    mex('-outdir', fullfile(fileparts(mfilename('fullpath')), 'private'), ...
        '-L', lib_path, ...
        '-ldpb_ffi', ...
        fullfile(fileparts(mfilename('fullpath')), 'private', 'dpb_mex.c'));

    fprintf('MEX file built successfully.\n');
end
