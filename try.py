import spxtacular as spx

r = spx.DReader("/home/patrick-garrett/Repos/tdfpy/tests/data/200ngHeLaPASEF_1min.d")
print(r.analysis_tdf_path)
print(r.analysis_dir)
print(r.analysis_tdf_bin_path)
for s in r.ms2:
    s = s.denoise()
    s = s.deconvolute()
    print(s.to_dict())
    break

