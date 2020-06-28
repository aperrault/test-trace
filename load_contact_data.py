import numpy as np
import pandas as pd

raw_participants = pd.read_csv('2008_Mossong_POLYMOD_participant_common.csv')

raw_contacts = pd.read_csv('2008_Mossong_POLYMOD_contact_common.csv')

contact_durations = raw_contacts['duration_multi'].to_numpy()
contact_durations = contact_durations[~np.isnan(contact_durations)]
