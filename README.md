# This will be spectacular
# A peptacular companion for handling spectra and such (with optional plotting)
# will use numpy
# will also use tdfpy and pymzml

```python
import spectacular as st


mzml_file = "blaaaa"
reader = st.reader(mzml_file)
spec = reader[0] # first spectra

or 

dfolder = "blass"
reader = st.reader(dfolder)
spec = reader[0] # first spectra

spec_plotter

    ions: tuple[str, set(int)]

    add_ion(ion_type, charge)

    # plotly 
    plot(title: str)
    


@frozen
peak:
    mz
    int
    charge
    im

@frozen
spec
    mz_arr # 1d
    intensity_arr #2d
    charge_array | None #3d
    im_arr | None #4d

    def deconvolute() -> self
        pass

    def peaks -> list[Peak]

    def top_peaks(n, by='intensity', reverse=False) -> list[Peak]

    @cached_property
    def _argsort_mz -> list[int]
    
    @cached_property
    def _argsort_intensity -> list[int]

    @cached_property
    def _argsort_charge -> list[int]

    @cached_property
    def _argsort_im_ -> list[int]

    def has_peak(target_mz, mz_tol, mz_tol_type, match_charg: bool, target_im | None, im_tol, im_tol_type) -> Bool

    def get_peak(target_mz, mz_tol, mz_tol_type, match_charg: bool, target_im | None, im_tol, im_tol_type, collision: Literal[largest, closest]) -> Peak

    def get_peaks(target_mz, mz_tol, mz_tol_type, match_charg: bool, target_im | None, im_tol, im_tol_type) -> List[Peak]

    def filter(min_mz, max_mz, min_int, max_int, ...) -> Self

    # plot annotate spec
    def annotate(peptide, ion_types, loss, isotopes) -> plot

    # plot spec
    def plot(title) -> plot
        pass

    

```# spextacular
