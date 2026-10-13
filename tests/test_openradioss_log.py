"""OpenRadioss engine-log energy error reader on a synthetic log."""
import math

from pinneapple_simulation.external_solvers.openradioss import read_energy_error, read_energy_errors

LOG = """\
 CYCLE    TIME   TIME-STEP  ELEMENT   ERROR
     0  0.000E+00  1.0E-06   SHELL  ERR=   0.0%
   100  1.000E-04  1.0E-06   SHELL  ERR=  -0.4%
   200  2.000E-04  1.0E-06   SHELL  ERR= -2.75%
   300  3.000E-04  1.0E-06   SHELL  ERR=   1.5%
 NORMAL TERMINATION
"""


def test_reads_series_and_max_abs(tmp_path):
    case = tmp_path / "Exp_1"
    case.mkdir()
    (case / "engine.log").write_text(LOG)
    assert read_energy_errors(case).tolist() == [0.0, -0.4, -2.75, 1.5]
    assert read_energy_error(case) == 2.75                    # largest |ERR|, sign dropped
    assert read_energy_error(case / "engine.log") == 2.75     # a log path works too


def test_no_err_lines_is_nan(tmp_path):
    log = tmp_path / "engine.log"
    log.write_text("ERROR TERMINATION\n")
    assert math.isnan(read_energy_error(log))
