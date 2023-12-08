import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def distance(start, end):
    return np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)

def remove_intervention(gps_data: pd.DataFrame, phase_data: pd.DataFrame, subject: int):
    """
    At first check whether intervention happened,
    secondly change end of phase such that end point is closest to starting point.
    """
    
    phase_data_new = phase_data.copy()
    gps_data_new = gps_data.copy()

    gps_data_new.rename({'NP_LatDegree': 'lat', 'NP_LongDegree': 'long'}, axis=1, inplace=True)

    for idx, phase in phase_data_new.iterrows():
        gps_data_new.loc[((gps_data_new.index > phase['start']) & (gps_data_new.index <= phase['end'])), 
                 ['phase', 'scenario']] = [phase['phase'], phase['scenario']]
    gps_data_new.dropna(inplace=True)
    
    end_loc = phase_data_new.columns.get_loc('end')
    
    for ids, scenario in enumerate(['city', 'rural', 'highway']):
        for idp, phase in enumerate([1, 2, 3, 4]):
            sub_phase_data = phase_data_new[(phase_data_new['phase'] == phase) & (phase_data_new['scenario'] == scenario)]
            if len(sub_phase_data) == 0:
                continue
            elif sub_phase_data.intervention.item():
                #print('Change end point for subject', subject, 'phase', phase, 'scenario', scenario)
        
                sub_gps_data = gps_data_new[(gps_data_new['phase'] == phase) & (gps_data_new['scenario'] == scenario)]
            
                if len(sub_gps_data) == 0:
                    continue
                    
                # take last part of gps data
                if (subject == 214) & ((phase == 3) & (scenario == 'rural') | (phase == 4) & (scenario == 'city')):
                    index_final = 60
                elif (subject == 219) & (phase == 2) & (scenario == 'highway'):
                    index_final = 100
                else:
                    index_final = int(len(sub_gps_data)/6.5)
                sub_gps_data_final = sub_gps_data[-index_final:].copy()
                
                # search for new end point which is closest to starting point
                start = [sub_gps_data.long[0], sub_gps_data.lat[0]]
                sub_gps_data_final['distance'] = np.nan
                dist_loc = sub_gps_data_final.columns.get_loc('distance')
                for i in range(len(sub_gps_data_final)):
                    end = [sub_gps_data_final.long[i], sub_gps_data_final.lat[i]]
                    sub_gps_data_final.iloc[i, dist_loc] = distance(start, end)
                min_loc = sub_gps_data_final.distance.idxmin()

                row_loc = phase_data_new[(phase_data_new['phase'] == phase) & (phase_data_new['scenario'] == scenario)].index[0]
                phase_data_new.iloc[row_loc, end_loc] = str(min_loc)
                
    return phase_data_new
