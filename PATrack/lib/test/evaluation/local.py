from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.network_path = ''    # Where tracking networks are stored.
    settings.prj_dir = ''
    settings.result_plot_path = ''
    settings.results_path = '/'    # Where to store tracking results
    settings.save_dir = ''
    settings.rgbd1k_path = ''
    settings.depthtrack_path = ''
    settings.show_results = False
    return settings

