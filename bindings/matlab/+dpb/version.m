function v = version()
%VERSION Get DPB library version
%   V = DPB.VERSION() returns the version string of the DPB library.
    v = dpb_mex('version');
end
