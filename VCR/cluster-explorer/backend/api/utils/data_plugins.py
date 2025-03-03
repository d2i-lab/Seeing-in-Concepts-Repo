import warnings
import numpy as np

class Plugin:
    def setup(self, df, hdf5):
        pass

    def cleanup(self, df):
        pass

    def filter(self, df, arg):
        pass


class PluginCrowding(Plugin):
    def setup(self, df, hdf5):
        gt_crowding = np.array(hdf5['gt_crowds'])
        dt_crowding = np.array(hdf5['pred_crowds'])
        df['crowding'] = np.maximum(gt_crowding, dt_crowding)
        return df

    def filter(self, df, crowding_threshold):
        current_size = len(df)
        df = df[df['crowding'] < crowding_threshold]
        print('Crowding filter: {} -> {} ({}%)'.format(
            current_size, len(df), len(df) / current_size * 100)
        )
        return df

    def cleanup(self, df):
        df = df.drop(columns='crowding')
        return df

class PluginConfusion(Plugin):
    def setup(self, df, hdf5):
        gt_confusion = np.array(hdf5['gt_confusions'])
        pred_confusion = np.array(hdf5['pred_confusions'])
        df['confusion'] = np.maximum(gt_confusion, pred_confusion)
        return df

    def filter(self, df, confusion_threshold):
        current_size = len(df)
        df = df[df['confusion'] < confusion_threshold]
        print('Confusion filter: {} -> {} ({}%)'.format(
            current_size, len(df), len(df) / current_size * 100)
        )
        return df

    def cleanup(self, df):
        df = df.drop(columns='confusion')
        return df

class PluginIgnoreFlag(Plugin):
    def setup(self, df, hdf5):
        ignore_flag = np.array(hdf5['ignore_flags'])
        df['ignore_flag'] = ignore_flag
        return df

    def filter(self, df, flag_value):
        print(flag_value)
        flag_value_set = set(int(f) for f in flag_value)

        flag_mask = df['ignore_flag'].isin(flag_value_set)

        current_size = len(df)
        df = df[flag_mask]
        print('Ignore flag filter: {} -> {} ({}%)'.format(
            current_size, len(df), len(df) / current_size * 100)
        )
        return df

    def cleanup(self, df):
        df = df.drop(columns='ignore_flag')
        return df

class PluginLabel(Plugin):
    def setup(self, df, hdf5):
        warnings.warn('Label plugin forces both gt and pred to be the same')
        return df

    def filter(self, df, label_value):
        current_size = len(df)
        df = df[df['gt_label'] == label_value]
        df = df[df['pred_label'] == label_value]
        
        print('Label filter: {} -> {} ({}%)'.format(
            current_size, len(df), len(df) / current_size * 100)
        )
        return df

    def cleanup(self, df):
        return df

    
PLUGINS = {
    'crowding': PluginCrowding(),
    'confusion': PluginConfusion(),
    'ignore_flag': PluginIgnoreFlag(),
    'label': PluginLabel()
}
