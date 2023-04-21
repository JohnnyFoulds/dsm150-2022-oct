import numpy as np
import pandas as pd

from typing import Dict, Optional

# the source data types
source_dtype = {
    "session_id": np.int64,
    "elapsed_time": np.int32,
    "event_name": "category",
    "name": "category",
    "level": np.uint8,
    "page": np.float32,
    "room_coor_x": np.float32,
    "room_coor_y": np.float32,
    "screen_coor_x": np.float32,
    "screen_coor_y": np.float32,
    "hover_duration": np.float32,
    "text": "category",
    "fqid": "category",
    "room_fqid": "category",
    "text_fqid": "category",
    "fullscreen": "category",
    "hq": "category",
    "music": "category",
    "level_group": "category",
}

def read_csv(path:str,
             compression:Optional[str]=None,
             preview_n:int=3,
             dtype:Dict[str, str]=None) -> pd.DataFrame:
    """Read the source data from a CSV file.
    
    Parameters
    ----------
    path : str
        The path to the CSV file.
    compression : str, optional
        The compression type of the CSV file, by default None
    preview_n : int, optional
        The number of rows to preview, by default 3
    dtype : Dict[str, str], optional
        The data types of the columns, by default None

    Returns
    -------
    pd.DataFrame
        The source data.
    """
    # load the source data
    df_data =  pd.read_csv(path,
                           dtype=dtype,
                           compression=compression)
    
    # show the data summary
    print(df_data.shape)
    with pd.option_context('display.max_columns', None):
        display(df_data.head(preview_n))

    return df_data
