from mantid import plots  # noqa F401
import matplotlib.pyplot as plt
from ornl.sans.sns import eqsans

# set up the configuration for preparing data
config = dict()

# files
config['mask'] = '/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/beamstop60_mask_4m.nxs'
config['flux'] = '/SNS/EQSANS/shared/instrument_configuration/bl6_flux_at_sample'
config['sensitivity_file_path'] \
    = '/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017A_mp/Sensitivity_patched_thinPMMA_4m_79165_event.nxs'
config['dark_current'] = '/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/EQSANS_86275.nxs.h5'

# numeric values
config['low_tof_clip'] = 500
config['high_tof_clip'] = 2000
config['detector_offset'] = 0.
config['sample_offset'] = 0.
config['bin_width'] = 0.5

# find the beam center
empty_fn = 'EQSANS_88973'
db_ws = eqsans.load_events(empty_fn)
center = eqsans.center_detector(db_ws)
config['center_x'] = center[0]
config['center_y'] = center[1]

# load and prepare scattering data
sample_file = "EQSANS_88980"
ws = eqsans.prepare_data(sample_file, **config)

absolute_scale = 0.0208641883
sample_thickness = 0.1

# background
bkg_fn = "EQSANS_88978"
bkg__trans_fn = "EQSANS_88974"

ws_bck = eqsans.prepare_data(bkg_fn, **config)

ws = eqsans.subtract_background(ws, background=ws_bck)

ws /= sample_thickness
ws *= absolute_scale

# If frame_skipping we will have more than one table workspace
table_ws_list = eqsans.prepare_momentum_transfer(ws, wavelength_binning=[config['bin_width']])

outputFilename = 'EQSANS_88980'
fig, ax = plt.subplots(subplot_kw={'projection': 'mantid'})
for index, table_ws in enumerate(table_ws_list):

    # TODO check the configuration-numQbins and configuration_QbinType
    numQBins = 100

    iq_ws = eqsans.cal_iq(table_ws, bins=numQBins, log_binning=False)

    suffix = '.txt'
    if len(table_ws_list) > 1:
        suffix = '_frame_{}{}'.format(index+1, suffix)
    outfile = outputFilename + suffix

    if index == 0:
        ax.plot(iq_ws)
        ax.set_yscale('log')

    eqsans.save_ascii_1D(iq_ws, outputFilename + suffix, outfile)
fig.savefig(outputFilename + '.png')
