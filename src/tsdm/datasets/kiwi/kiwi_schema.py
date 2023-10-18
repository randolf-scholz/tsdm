"""Schema for the KIWI-dataset."""

# NOTE: THis should only contain static data

__all__ = ["timeseries_description", "metadata_description"]

# fmt: off
timeseries_description = {
    # Name                          : [Unit,     Type,  Dtype, Lower, Upper, Lower included, Upper included]
    "Acetate"                       : ["%",      "percent",  "float32", 0,   100,       True, True],
    "Base"                          : ["uL",     "absolute", "float32", 0,   None,      True, True],
    "Cumulated_feed_volume_glucose" : ["uL",     "absolute", "float32", 0,   None,      True, True],
    "Cumulated_feed_volume_medium"  : ["uL",     "absolute", "float32", 0,   None,      True, True],
    "DOT"                           : ["%",      "percent",  "float32", 0,   100,       True, True],
    "Flow_Air"                      : ["Ln/min", "absolute", "float32", 0,   None,      True, True],
    "Fluo_GFP"                      : ["RFU",    "absolute", "float32", 0,   1_000_000, True, True],
    "Glucose"                       : ["g/L",    "absolute", "float32", 0,   20,        True, True],
    "InducerConcentration"          : ["mM",     "absolute", "float32", 0,   None,      True, True],
    "OD600"                         : ["%",      "percent",  "float32", 0,   100,       True, True],
    "Probe_Volume"                  : ["uL",     "absolute", "float32", 0,   None,      True, True],
    "StirringSpeed"                 : ["U/min",  "absolute", "float32", 0,   None,      True, True],
    "Temperature"                   : ["°C",     "linear",   "float32", 20,  45,        True, True],
    "Volume"                        : ["mL",     "absolute", "float32", 0,   None,      True, True],
    "pH"                            : ["pH",     "linear",   "float32", 4,   10,        True, True],
}
# fmt: on

# fmt: off
metadata_description = {
    # Name                   : [Unit,  Type,   Dtype, Lower, Upper, Lower included, Upper included]
    "Feed_concentration_glc" : ["g/L", "absolute", None, None, None, True, True],
    "IPTG"                   : ["mM",  "absolute", None, 0,    None, True, True],
    "OD_Dilution"            : ["%",   "percent",  None, 0,    100,  True, True],
    "bioreactor_id"          : [None,  "category", None, None, None, True, True],
    "color"                  : [None,  "category", None, None, None, True, True],
    "container_number"       : [None,  "category", None, None, None, True, True],
    "pH_correction_factor"   : [None,  "factor",   None, 0,    None, True, True],
    "ph_Tolerance"           : [None,  "linear",   None, 0,    None, True, True],
    "plasmid_id"             : [None,  "category", None, None, None, True, True],
    "profile_name"           : [None,  "category", None, None, None, True, True],
    "μ_set"                  : ["%",   "percent",  None, 0,    100,  True, True],
}
# fmt: on
