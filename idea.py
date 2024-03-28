from NEAST_models.BBP.neat_model import NeatModel, BBPConfig
from pyrmodel import pyrconfig

# Load BBP-L2/3-pyramidal cell as NeatModel
channels_pyr = [
    "Ca_HVA",
    "Ca_LVAst",
    "Ih",
    "K_Tst",
    "K_Pst",
    "Im",
    "NaTa_t",
    "NaTs2_t",
    "SKv3_1",
    "SK_E2",
    "Nap_Et2",
]
model = NeatModel(BBPConfig(**pyrconfig), channels=channels_pyr, w_ca_conc=True, passified_dendrites=False)

# Plot cell

# Simulate basic stimulation


# Load same model but passified version
model = NeatModel(BBPConfig(**pyrconfig), channels=channels_pyr, w_ca_conc=True, passified_dendrites=True)


# Simulate basic stimulation
